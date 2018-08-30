from cofe_files.scripts.create_graph import run_create_graph
from chan_modeling.archiving import MyThreads
from cofe_files.scripts.expand import run_expand

N_ADJS = 10
N_TIMES = 10

def expand_archive(archive_folder_name):
    """
    Run the process to expand a folder of threads
    :param archive_folder_name:
    :return:
    """
    threads = MyThreads(archive_folder_name)
    graph_file_name = run_create_graph(threads=threads,
                         nAdjs=N_ADJS,
                         filename='my_threads.txt',
                         outname='my_threads')
    run_expand(threads, 'expanded_threads.txt', graph_file_name, N_TIMES)


