"""Tests for ledger.ascend_import: Ascend CANN msprof + aclGraph import and end-to-end Ledger analysis."""

import json
import sys

import pytest

from ledger import analyze, ascend_import as ascend, cli, perfetto, transfers
from report import section, table_rows


def _write_msprof_fixture(root):
    """Build a minimal Ascend CANN msprof directory with AIV-expanded and SDMA collectives."""
    out = root / "PROF_000001_test" / "mindstudio_profiler_output"
    out.mkdir(parents=True)

    # Op summary:
    # - 1 weight-streaming INT8 matmul on stream 33, task 10 (matched by aclGraph)
    # - 1 draft BF16 matmul on stream 27, task 5 (matched via --stream-map 27:22)
    # - 1 paired AIV collective (Stream ID N/A summary + device AivKernel on stream 31, task 1)
    # - 1 unmatched SDMA/RDMA collective (Stream ID N/A only -> kept on virtual stream -1)
    op_summary = "\n".join([
        "Device_id,Model ID,Task ID,Stream ID,Op Name,OP Type,OP State,Task Type,Task Start Time(us),Task Duration(us),Task Wait Time(us),Block Num,Mix Block Num",
        "0,4294967295,10,33,QuantBatchMatmulV3_int8,QuantBatchMatmulV3,N/A,MIX_AIC,1000.000\t,20.000,0,16,0",
        "0,48,N/A,N/A,hcom_allGather__503_1_1,hcom_allGather,N/A,COMMUNICATION,1025.000\t,15.000,0,N/A,N/A",
        "0,48,1,31,AivKernel,AivKernel,N/A,COMMUNICATION,1025.000\t,15.000,0,8,0",
        "0,4294967295,N/A,N/A,hcom_reduceScatter__503_2_1,hcom_reduceScatter,N/A,COMMUNICATION,1045.000\t,30.000,0,N/A,N/A",
        "0,4294967295,5,27,MatMulV2_bf16,MatMulV2,N/A,AI_CORE,1080.000\t,40.000,0,24,0",
    ]) + "\n"
    (out / "op_summary_20261127.csv").write_text(op_summary)

    task_time = "\n".join([
        "Device_id,kernel_name,kernel_type,stream_id,task_id,task_time(us),task_start(us),task_stop(us)",
        "0,N/A,MEMCPY_ASYNC,43,100,5.000,1040.000\t,1045.000\t",
        "0,QuantBatchMatmulV3_int8,MIX_AIC,33,10,20.000,1000.000\t,1020.000\t",
    ]) + "\n"
    (out / "task_time_20261127.csv").write_text(task_time)

    w_addr = 0x12D0_0000_0000
    d_addr = 0x12E0_0000_0000
    c_addr = 0x12F0_0000_0000

    capture_graph = [
        {
            "name": "QuantBatchMatmulV3_int8",
            "args": {
                "Model Id": 48,
                "Stream Id": 33,
                "Task Id": 10,
                "Numblocks": 16,
                "Kernel Args": f"0x{w_addr:016x} 0x{c_addr:016x}",
            },
        },
        {
            "name": "aiv_all_gather_bfloat16_t",
            "args": {
                "Model Id": 48,
                "Stream Id": 31,
                "Task Id": 1,
                "Numblocks": 8,
                "Kernel Args": "0x000012c000001000 0x0000000000000008",
            },
        },
    ]
    draft_graph = [
        {
            "name": "MatMulV2_bf16",
            "args": {
                "Model Id": 46,
                "Stream Id": 22,
                "Task Id": 5,
                "Numblocks": 24,
                "Kernel Args": f"0x{d_addr:016x}",
            },
        }
    ]
    target_weights = {
        "model.layers.0.self_attn.wq_a.weight": {
            "dtype": "torch.int8",
            "shape": [4096, 1024],
            "address": w_addr,
            "bytes": 4194304,
        }
    }
    draft_weights = {
        "model.layers.43.self_attn.wo_b.weight": {
            "dtype": "torch.bfloat16",
            "shape": [4096, 2048],
            "address": d_addr,
            "bytes": 16777216,
        }
    }
    owners = {
        "model.layers.0.self_attn.kv_cache": [
            {
                "dtype": "torch.bfloat16",
                "shape": [1024, 512],
                "address": c_addr,
                "bytes": 1048576,
            }
        ]
    }

    (root / "capture.json").write_text(json.dumps(capture_graph))
    (root / "draft.json").write_text(json.dumps(draft_graph))
    (root / "target_weights.json").write_text(json.dumps(target_weights))
    (root / "draft_weights.json").write_text(json.dumps(draft_weights))
    (root / "owners.json").write_text(json.dumps(owners))
    return root


def test_ascend_import_and_end_to_end_analysis(tmp_path, monkeypatch):
    """Importing an Ascend msprof + aclGraph capture feeds analyze, transfers, and perfetto cleanly."""
    src = _write_msprof_fixture(tmp_path / "src")
    cap = tmp_path / "capture"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ledger",
            "ascend-import",
            str(src),
            str(cap),
            "--graph", str(src / "capture.json"),
            "--graph", str(src / "draft.json"),
            "--stream-map", "27:22",
            "--weights", str(src / "target_weights.json"),
            "--draft-weights", str(src / "draft_weights.json"),
            "--owners", str(src / "owners.json"),
            "--device", "Ascend 910B1",
            "--sms", "24",
            "--steps", "1",
            "--batch", "8",
            "--draft-tokens", "7",
        ],
    )
    cli.main()

    run = json.loads((cap / "run.json").read_text())
    assert run["device"] == "Ascend 910B1"
    assert run["backend"] == "ascend"
    assert run["sms"] == 24

    kernels = [json.loads(line) for line in (cap / "kernels-0.jsonl").read_text().splitlines()]
    # Paired AIV collective is deduplicated (4 total kernels, not 5), and unmatched SDMA collective is retained.
    assert len(kernels) == 4
    names = [k["name"] for k in kernels]
    assert "hcom_allGather__503_1_1:aiv_all_gather_bfloat16_t" in names
    assert "hcom_reduceScatter__503_2_1" in names
    sdma = [k for k in kernels if k["name"] == "hcom_reduceScatter__503_2_1"][0]
    assert sdma["stream"] == -1

    report_path = tmp_path / "report.md"
    analyze.report(str(cap), ceiling=1350, steps=1, out=str(report_path),
                   peaks=analyze.parse_peaks("bf16=320,int8=640"), sms=24)
    text = report_path.read_text()

    weight_rows = {r[0].strip("`"): r for r in table_rows(section(text, "Weight-streaming launches by weight"))}
    assert "model.layers.N.self_attn.wq_a.weight" in weight_rows
    assert "draft.model.layers.N.self_attn.wo_b.weight" in weight_rows

    compute_rows = {r[0].strip("`"): r for r in table_rows(section(text, "Compute"))}
    assert compute_rows["model.layers.N.self_attn.wq_a.weight"][1] == "int8"
    assert compute_rows["model.layers.N.self_attn.wq_a.weight"][8] == "memory"

    # Transfers picks up MEMCPY_ASYNC from task_time_*.csv (5 us exposed between 1040..1045 us).
    xfer_path = tmp_path / "transfers.md"
    with open(xfer_path, "w") as f:
        transfers.report(str(cap), {}, steps=1, out=f)
    assert "5.0 us of it with no kernel running" in xfer_path.read_text()

    # Perfetto export produces valid Chrome Trace JSON.
    trace_path = tmp_path / "trace.json"
    perfetto.export(str(cap), ceiling=1350, out=str(trace_path))
    trace = json.loads(trace_path.read_text())
    assert len(trace["traceEvents"]) > 4


def test_parse_kernel_args_ignores_non_hex_tokens():
    params = ascend.parse_kernel_args("0x000012c07ca00000 not_hex 0x10")
    assert len(params) == 1
    off, size, raw_hex = params[0]
    assert off == 0 and size == 16
    assert int.from_bytes(bytes.fromhex(raw_hex)[:8], "little") == 0x12C07CA00000
    assert int.from_bytes(bytes.fromhex(raw_hex)[8:16], "little") == 0x10


def test_import_capture_missing_op_summary_exits(tmp_path):
    with pytest.raises(SystemExit, match="op_summary"):
        ascend.import_capture(str(tmp_path), str(tmp_path / "out"))


def test_timestamps_keep_nanoseconds():
    assert ascend._parse_us_ns("1764251755123456.789\t") == 1764251755123456789


def test_clashing_graph_dumps_warn(tmp_path, capsys):
    paths = []
    for name, arg in (("a", "0x10"), ("b", "0x20"), ("c", "0x20")):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps([{"name": name, "args": {"Stream Id": 1, "Task Id": 1, "Kernel Args": arg}}]))
        paths.append(str(path))
    assert ascend.load_acl_graphs(paths)[(1, 1)]["name"] == "c"
    assert "1 (stream, task) pairs" in capsys.readouterr().err


def test_bad_stream_map_exits():
    with pytest.raises(SystemExit, match="--stream-map"):
        ascend._parse_stream_maps(["23"])
