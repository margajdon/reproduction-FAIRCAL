import os


def generate_file_structure():

	os.system("tree /a > filestructure.txt")

	lines = open("filestructure.txt", "r").readlines()
	lines = lines[1:]
	lines[0] = lines[0][:6]
	lines = [line for line in lines if line.count(" ") < 12]
	lines = [line.strip() for line in lines if len(line.strip()) > 3]
	lines = [line for line in lines if not line.endswith(".pyc")]
	lines = [line for line in lines if "pycache" not in line]

	with open("filestructure.txt", "w") as f:
		f.write("\n".join(lines))

if __name__ == "__main__":
	generate_file_structure()
