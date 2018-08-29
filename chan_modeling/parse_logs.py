import re
import sys
import pprint

def get_perplexity_series_from_logs(log_file_name):
	matches = []
	with open(log_file_name, 'r') as f:
		for line in f:
			iteration_matches = re.findall(r'([0-9]+) topics, 100 passes over the supplied corpus of ([0-9]+) documents', line)
			perplexity_matches = re.findall(r'([0-9\.]+) perplexity estimate', line) 
			if iteration_matches:
				matches.append(iteration_matches)
			if perplexity_matches:
				matches.append(perplexity_matches)

		filtered_matches = []
		for i in range(len(matches)):
			if type(matches[i][0]) == tuple:
				new_data = list(matches[i][0])
				new_data.append(matches[i-1][0])
				filtered_matches.append(new_data)

		one = filtered_matches[0::3]
		two = filtered_matches[1::3]
		three = filtered_matches[2::3]
		return one, two, three
