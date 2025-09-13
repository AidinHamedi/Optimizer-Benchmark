sync_docs:
	python tools/sync_docs.py

gen_docs:
	python tools/md_comparison.py
	python tools/doc_visualizations.py

comp_vis:
	tar -czf results.tar.gz results/

clear:
	rm -r cache/
