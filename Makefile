PYTHON ?= python3
EXP ?= EXP-20261008-101
OUT ?= results/runs/$(EXP)
BACKENDS = pytorch,torch_compile,cuda,triton,cute,cublas,cublaslt,cuda_fast,triton_fast,cute_fast

.PHONY: test evidence-check list smoke benchmark w4a16-smoke w4a16-benchmark profile-preview
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

profile-preview:
	$(PYTHON) scripts/profile_comparison.py --experiment $(EXP) --output results/collected/$(EXP)/fp32 --optimized
	$(PYTHON) scripts/profile_w4a16.py --experiment $(EXP) --output results/collected/$(EXP)/w4a16
