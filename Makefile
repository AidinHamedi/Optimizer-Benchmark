sync_docs:
	cp README.md docs/README.md

gen_docs:
	python tools/md_comparison.py
	python tools/md_visualizations.py

comp_vis:
	tar -czf results.tar.gz results/

clear:
	rm -r cache/
