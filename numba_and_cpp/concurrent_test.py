
from concurrent.futures import ThreadPoolExecutor
import example

print(example.add(1, 4))

print('starting 1')

example.function_that_takes_a_while()

with ThreadPoolExecutor(4) as ex:
	ex.map(lambda x: example.function_that_takes_a_while(), [None] * 4)  # function_that_takes_a_while is from example.cpp

print('ending 2')



