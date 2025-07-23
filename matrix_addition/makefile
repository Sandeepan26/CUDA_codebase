automatic: compile run

compile:
	nvcc *.cu -o code.out && echo "Compilation Successful"

run:
	./code.out && echo "Successful run"

analyze:
	ncu ./code.out

clean:
	rm -f *.out