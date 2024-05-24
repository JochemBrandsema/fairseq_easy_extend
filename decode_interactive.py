from torch.profiler import profile, record_function
from fairseq_easy_extend_cli import interactive

if __name__ == "__main__":
    with profile(profile_memory=True) as prof:
        with record_function("decoding"):
            interactive.cli_main()
    print(prof.key_averages())
