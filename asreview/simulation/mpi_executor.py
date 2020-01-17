from mpi4py import MPI


def mpi_worker(job_runner):
    comm = MPI.COMM_WORLD
    while True:
        job = comm.recv(source=0)
        if job is None:
            break

        job_runner.execute(**job)
        comm.send(None, dest=0)
    return None, None


def mpi_executor(all_jobs, job_runner=None, server_job=True):
    comm = MPI.COMM_WORLD
    n_proc = comm.Get_size()

    for i_proc in range(1, n_proc):
        try:
            job = all_jobs.pop()
        except IndexError:
            break

        comm.send(job, dest=i_proc)

    if server_job:
        try:
            job = all_jobs.pop()
            job_runner.execute(**job)
        except IndexError:
            pass

    n_jobs_sent = 0
    while len(all_jobs) > 0:
        job = all_jobs.pop()
        if server_job and (n_jobs_sent % n_proc) == n_proc - 1:
            job_runner.execute(**job)
            n_jobs_sent += 1
            continue

        status = MPI.Status()
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        pid = status.source
        comm.send(job, dest=pid)
        n_jobs_sent += 1

    for i_proc in range(1, n_proc):
        comm.recv(source=MPI.ANY_SOURCE)


def mpi_hyper_optimize(job_runner, n_iter):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        job_runner.hyper_optimize(n_iter)
    else:
        mpi_worker(job_runner)
