def serial_executor(jobs, job_runner):
    for job in jobs:
        job_runner.execute(**job)


def serial_hyper_optimize(job_runner, n_iter):
    job_runner.hyper_optimize(n_iter)
