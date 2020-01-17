from mpi4py import MPI
# from asreview.simulation.parameter_opt import run_model #, compute_args
# from asreview.analysis import Analysis
# from hyperopt.base import STATUS_OK
# from numpy import average


def mpi_worker(job_function):
    comm = MPI.COMM_WORLD
    while True:
        print(f"Job {comm.Get_rank()}: Start receiving jobs")
        data = comm.recv(source=0)
        print(f"Job {comm.Get_rank()}: received job.")
        if data is None:
            print(f"Job {comm.Get_rank()}: Job queue finished.")
            break

#         print(f"Job {comm.Get_rank()}: received job.")
        run_args = data['run_args']
        run_kwargs = data['run_kwargs']
        job_function(*run_args, **run_kwargs)
        print(f"Job {comm.Get_rank()}: finished job.")
        comm.send(None, dest=0)
    return None, None


# def create_jobs(files, *args, n_runs=8, **kwargs):
#     jobs = []
#     log_dirs = []
#     for dataname, data_fp in files.items():
#         for i_run in range(n_runs):
#             run_args, run_kwargs, log_dir = compute_args(
#                 dataname, data_fp, *args, **kwargs, i_run=i_run)
#             jobs.append({"run_args": run_args, "run_kwargs": run_kwargs})
#             log_dirs.append(log_dir)
#     return jobs, log_dirs


def mpi_server(job_data, job_function, server_job=True):
    comm = MPI.COMM_WORLD
    n_proc = comm.Get_size()

#     jobs, log_dirs = create_jobs(*args, **kwargs)
    print(f"Server: {len(job_data)} jobs, {n_proc} instances.")
    for i_proc in range(1, n_proc):
        try:
            job = job_data.pop()
        except IndexError:
            break

        print(f"Send job to proc {i_proc}")
        comm.send(job, dest=i_proc)

    if server_job:
        try:
            job = job_data.pop()
            job_function(*job['run_args'], **job['run_kwargs'])
        except IndexError:
            pass

    n_jobs_sent = 0
    while len(job_data) > 0:
        print(f"Server: {len(job_data)} to go.")
        job = job_data.pop()
        if server_job and (n_jobs_sent % n_proc) == n_proc - 1:
            job_function(*job['run_args'], **job['run_kwargs'])
            n_jobs_sent += 1
            continue

        status = MPI.Status()
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        pid = status.source
        comm.send(job, dest=pid)
        n_jobs_sent += 1

    for i_proc in range(1, n_proc):
        print(f"Wait for finish {i_proc}")
        comm.recv(source=MPI.ANY_SOURCE)

#     losses = []
#     for log_dir in log_dirs:
#         analysis = Analysis.from_dir(log_dir)
#         ttd = analysis.avg_time_to_discovery()
#         losses.append(loss_spread(ttd, len(analysis.labels), 1.0))

#     return {"loss": average(losses), 'status': STATUS_OK}


def mpi_hyper_optimize(optimize_func, job_function, *args, server_job=True, **kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        data = optimize_func(*args, executor=mpi_server, **kwargs)
        for pid in range(1, comm.Get_size()):
            comm.send(None, dest=pid)
        return data
    else:
        return mpi_worker(job_function)
