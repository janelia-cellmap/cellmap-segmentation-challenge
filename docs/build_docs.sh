sphinx-apidoc -o ./source ../src/
make clean
make html
sphinx-build -M coverage source build
