import collections

# 621
task = ["A", "A", "A", "B", "B", "C", "D", "E", "E", "E", "F", "F", "G", "I"]
N = 6

task_cnt = collections.Counter(task).values()
M = max(task_cnt)
temp = collections.Counter(task_cnt)
Mct = temp[M]

a = 1

c = collections.Counter(a=4, b=2, c=0, d=-2)
d = collections.Counter(b=2, c=3, d=4)
c.subtract(d)

a = 1