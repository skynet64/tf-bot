import subprocess


# Downloads reddit comments from archive

# Download range
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
    cmd = ['wget', '']
    t_link = r'http://files.pushshift.io/reddit/comments/RC_'

    for name in file_names(start_year, end_year):
        cmd[1] = t_link + name + '.bz2'
        handle = subprocess.Popen(args=cmd)
        handle.wait()
