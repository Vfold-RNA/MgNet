.PHONY: descriptor_ext

descriptor_ext:
	g++ -c -fPIC descriptor_ext.cpp -o descriptor_ext.o -std=c++11
	g++ -shared -Wl,-soname,libdescriptor_ext.so  -o libdescriptor_ext.so descriptor_ext.o -lpthread -std=c++11
	#python test.py
