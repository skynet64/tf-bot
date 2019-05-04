import os
import subprocess
import shutil


# Unpacks downloaded comment archives

# Unpack range
start_year = 2005
end_year = 2011


def file_names(begin, end):
    """Yields names in format YYYY-MM."""

    for year in range(begin, end):
        for month in range(1, 13):
            file_name = str(year) + '-'
            if month < 10:
                file_name += '0'
            file_name += str(month)
            yield file_name


if __name__ == '__main__':
    cmd = ['bzip2', '-d', '']
    t_name = 'RC_'

    for new_name in file_names(start_year, end_year):
        old_name = t_name + new_name
        cmd[2] = old_name + '.bz2'
        if not (os.path.isfile(cmd[2])):
            continue
        print('Unpacking ' + cmd[2])
        handle = subprocess.Popen(args=cmd)
        handle.wait()
        new_name += '.json'
        shutil.move(old_name, new_name)
