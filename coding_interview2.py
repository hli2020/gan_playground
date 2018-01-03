import collections

task = ["A", "A", "A", "B", "B", "C", "D", "E", "E", "E", "F", "F", "G", "I"]
N = 6

task_cnt = collections.Counter(task).values()
M = max(task_cnt)
temp = collections.Counter(task_cnt)
Mct = temp[M]

a = 1