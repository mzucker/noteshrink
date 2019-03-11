all: install \
	example_output/notesA.pdf \
	example_output/notesB.pdf \
	example_output/tree.pdf \
	example_output/graph-paper-ink-only.pdf

install:
	@echo Install PIP requirements
	pip3 install -r requirements.txt
	@echo Installation finished, install imagemagick manually

docker_build:
	@echo Building Docker container noteshrink
	docker build -t noteshrink -f Dockerfile .

example_output/notesA.pdf: examples/notesA1.jpg examples/notesA2.jpg
	mkdir -p example_output && \
	cd example_output && \
	../noteshrink.py -O -g -w -s 20 -v 30 -o notesA.pdf ../examples/notesA*.jpg

example_output/notesB.pdf: examples/notesB1.jpg examples/notesB2.jpg
	mkdir -p example_output && \
	cd example_output && \
	../noteshrink.py -O -g -w -s 4.5 -v 30 -o notesB.pdf ../examples/notesB*.jpg

example_output/tree.pdf: examples/tree.jpg
	mkdir -p example_output && \
	cd example_output && \
	../noteshrink.py -O -w -o tree.pdf ../examples/tree.jpg

example_output/graph-paper-ink-only.pdf: examples/graph-paper-ink-only.jpg
	mkdir -p example_output && \
	cd example_output && \
	../noteshrink.py -O -v 5 -o graph-paper-ink-only.pdf ../examples/graph-paper-ink-only.jpg
