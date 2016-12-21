f="all_tag.txt"

for line in open(f):
	str=line.replace(":)","").replace(":(","")
	print(str, end="")
