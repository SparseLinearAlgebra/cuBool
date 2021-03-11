export PYTHONPATH="`pwd`:$PYTHONPATH"
cd tests
python3 -m unittest discover -v
cd ..