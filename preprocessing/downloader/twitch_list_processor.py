f = open("a.txt","r")
for line in f:
	line = line.rstrip()
	
	if "92m" in line:
		if "Enter" not in line:
			dummy = line[:1]
			line =line.replace(dummy+"[92m","")
			line = line.replace(dummy+"[0m","")
			print(line)

	if "Published" in line:
		dummy = line[10:11]
		line = line.replace(dummy+"[94m","")
		line = line.replace(dummy+"[0m","")
		print(line)
	if "https" in line:
		dummy = line[:1]
		line = line.replace(dummy+"[3m","")
		line = line.replace(dummy+"[0m","")
		print(line)
