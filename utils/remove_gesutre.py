import os
import csv
import argparse

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Remove keypoinys with class_id")
parser.add_argument("-i",
                    "--class_id",
                    help="class_id you want to remove from keypoint.csv",
                    type=int, default=-1)

args = parser.parse_args()
curr_file_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.realpath(os.path.join(curr_file_dir, '..\\model\\gesture_classifier\\keypoint.csv'))


def main(remove_class_id):
    lines = []
    with open(data_file_path, 'r') as readFile:

        reader = csv.reader(readFile)

        for row in reader:
            # might contain empty row
            if row != [] and row[0] != str(remove_class_id):
                lines.append(row)

    with open(data_file_path, 'w') as writeFile:

        writer = csv.writer(writeFile)
        writer.writerows(lines)


if __name__ == '__main__':
    print(data_file_path)
    # print("Removing keypoints with class_id: {}".format(args.class_id))
    # main(args.class_id)
    # print("Complete!")
