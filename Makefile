# https://unix.stackexchange.com/a/100798
diff:
	diff data/raw/test.txt data/test.txt > data/diff_raw_test_to_test.txt; [ $$? -eq 1 ]
