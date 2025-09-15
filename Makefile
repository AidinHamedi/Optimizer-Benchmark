PYTHON := python
SCRIPT := runner.py
PROCS ?= 2

clear:
	@rm -r cache/

run:
	@if ! command -v tmux >/dev/null 2>&1; then \
		echo "Error: tmux is not installed."; \
		exit 1; \
	fi
	@TOTAL=$$($(PYTHON) $(SCRIPT) --get_num); \
	if [ $$TOTAL -le 0 ]; then \
		echo "No optimizers found."; \
		exit 1; \
	fi; \
	echo "Total optimizers: $$TOTAL"; \
	CHUNK_SIZE=$$(( ($$TOTAL + $(PROCS) - 1) / $(PROCS) )); \
	echo "Running in $(PROCS) tmux panes, chunk size: $$CHUNK_SIZE"; \
	tmux new-session -d "$(PYTHON) $(SCRIPT) --range 0 $$CHUNK_SIZE"; \
	START=$$CHUNK_SIZE; \
	for i in $$(seq 2 $(PROCS)); do \
		END=$$((START + CHUNK_SIZE)); \
		if [ $$END -gt $$TOTAL ]; then END=$$TOTAL; fi; \
		tmux split-window -h "$(PYTHON) $(SCRIPT) --range $$START $$END"; \
		tmux select-layout tiled; \
		START=$$END; \
	done; \
	tmux attach

gen_docs:
	@python tools/md_comparison.py
	@python tools/doc_visualizations.py

sync_docs:
	@python tools/sync_docs.py

comp_vis:
	@tar -czf results.tar.gz results/
