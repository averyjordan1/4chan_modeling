from cofe_files.scripts.create_graph import run_create_graph
from chan_modeling.archiving import MyThreads
from cofe_files.scripts.expand import run_expand

N_ADJS = 10
N_TIMES = 4

def expand_archive(archive_folder_name, specific_threads=None):
    """
    Run the process to expand a folder of threads
    :param archive_folder_name:
    :return:
    """
    if specific_threads:
        threads = specific_threads
    else:
        threads = MyThreads(archive_folder_name)
    graph_file_name = run_create_graph(threads=threads,
                         nAdjs=N_ADJS,
                         filename='my_threads.txt',
                         outname='my_threads')
    run_expand(threads, 'expanded_threads.txt', graph_file_name, N_TIMES)


