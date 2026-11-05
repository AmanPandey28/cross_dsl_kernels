PYTHON ?= python3
EXP ?= EXP-20261008-101
OUT ?= results/runs/$(EXP)
BACKENDS = pytorch,torch_compile,cuda,triton,cute,cublas,cublaslt,cuda_fast,triton_fast,cute_fast

.PHONY: test evidence-check list smoke benchmark w4a16-smoke w4a16-benchmark profile-preview bf16-check bf16-smoke bf16-benchmark bf16-profile-preview
.PHONY: gemv-list gemv-check gemv-benchmark gemv-profile-preview rmsnorm-compare
test:
	PYTHONPATH=src:scripts:benchmarks $(PYTHON) -m unittest discover -s tests/unit -v

evidence-check:
	$(PYTHON) scripts/check_evidence.py

list:
	$(PYTHON) benchmarks/compare.py --experiment $(EXP) --output $(OUT)/fp32 --list

smoke:
	$(PYTHON) benchmarks/compare.py --experiment $(EXP) --output $(OUT)/fp32 --smoke --backends $(BACKENDS) --warmup 2 --repeat 3 --graph-calls 2 --no-calibrate

benchmark:
	$(PYTHON) benchmarks/compare.py --experiment $(EXP) --output $(OUT)/fp32 --backends $(BACKENDS) --warmup 5 --repeat 20 --graph-calls 20

w4a16-smoke:
	$(PYTHON) benchmarks/w4a16_benchmark.py --experiment $(EXP) --output $(OUT)/w4a16 --mode check --smoke --warmup 2 --repeat 3 --graph-calls 2

w4a16-benchmark:
	$(PYTHON) benchmarks/w4a16_benchmark.py --experiment $(EXP) --output $(OUT)/w4a16 --mode benchmark --warmup 5 --repeat 20 --graph-calls 10

bf16-check:
	$(PYTHON) scripts/verify_rmsnorm_bf16.py --experiment $(EXP) --output $(OUT)/bf16-checks.json

bf16-smoke:
	$(PYTHON) benchmarks/rmsnorm_bf16.py --experiment $(EXP) --output $(OUT)/bf16 --cases bf16_3x513,bf16_256x4096 --rounds 2 --warmup 2 --graph-calls 2

bf16-benchmark:
	$(PYTHON) benchmarks/rmsnorm_bf16.py --experiment $(EXP) --output $(OUT)/bf16 --rounds 20 --warmup 10 --graph-calls 20

bf16-profile-preview:
	$(PYTHON) scripts/profile_rmsnorm_bf16.py --experiment $(EXP) --output results/collected/$(EXP)/bf16

gemv-list:
	$(PYTHON) benchmarks/gemv_cuda_benchmark.py --experiment $(EXP) --output $(OUT)/gemv --list

gemv-check:
	$(PYTHON) scripts/verify_gemv_simt.py --experiment $(EXP) --output $(OUT)/gemv-checks.json

gemv-benchmark:
	$(PYTHON) benchmarks/gemv_cuda_benchmark.py --experiment $(EXP) --output $(OUT)/gemv --warmup 5 --repeat 30 --graph-calls 30

gemv-profile-preview:
	$(PYTHON) scripts/profile_gemv_simt.py --experiment $(EXP) --output results/collected/$(EXP)/gemv

rmsnorm-compare:
	$(PYTHON) benchmarks/rmsnorm_cuda_compare.py --experiment $(EXP) --output $(OUT)/rmsnorm-paired --before results/snapshots/rmsnorm_before --rounds 20 --warmup 10 --graph-calls 20

profile-preview:
	$(PYTHON) scripts/profile_comparison.py --experiment $(EXP) --output results/collected/$(EXP)/fp32 --optimized
	$(PYTHON) scripts/profile_w4a16.py --experiment $(EXP) --output results/collected/$(EXP)/w4a16
