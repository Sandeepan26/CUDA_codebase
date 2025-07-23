automatic: compile run

compile:
	nvcc vectorAdd.cu -o vector_add_res && echo "Compilation Successful"

run:
	./vector_add_res && echo "Successful run"

analyze:
	ncu ./vector_add_res

clean:
	rm -f vector_add_res