export PYTHONPATH=":`pwd`$PYTHONPATH"
cd tests
python3 -m unittests discover -v
cd ..