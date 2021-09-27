# https://unix.stackexchange.com/a/100798
diff:
	diff data/raw/test.txt data/test.txt > data/diff_raw_test_to_test.txt; [ $$? -eq 1 ]
	wc data/raw/test.txt
	wc data/test.txt

data:
	cut -c 7- data/raw/train.txt > data/train.txt

stats:
	@echo "Number of abstracts:"
	grep ^AB data/raw/train.txt | wc -l
