OS := $(shell uname -s)

ifeq ($(OS), Linux)
  NPROCS := $(shell grep -c ^processor /proc/cpuinfo)
else ifeq ($(OS), Darwin)
  NPROCS := 2
else
  NPROCS := 0
endif # $(OS)

ifeq ($(NPROCS), 2)
	CONCURRENCY := 2
else ifeq ($(NPROCS), 1)
	CONCURRENCY := 1
else ifeq ($(NPROCS), 3)
	CONCURRENCY := 3
else ifeq ($(NPROCS), 0)
	CONCURRENCY := 0
else
	CONCURRENCY := $(shell echo "$(NPROCS) 2" | awk '{printf "%.0f", $$1 / $$2}')
endif

.PHONY: lint style black test test_ci coverage clean

all_check: style lint

lint:
	pylint -rn qopt_best_practices test

black:
	python -m black qopt_best_practices test

style:
	python -m black --check qiskit_algorithms test

test:
	python -m unittest discover -v test

test_ci:
	echo "Detected $(NPROCS) CPUs running with $(CONCURRENCY) workers"
	python -m stestr run --concurrency $(CONCURRENCY)

coverage:
	python -m coverage3 run --source qopt_best_practices -m unittest discover -s test -q
	python -m coverage3 report

coverage_erase:
	python -m coverage erase

clean: coverage_erase;
