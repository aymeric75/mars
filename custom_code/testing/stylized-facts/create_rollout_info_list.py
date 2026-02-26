from pathlib import Path
from market_simulation.utils import pkl_utils
from report_stylized_facts import RolloutInfo, get_rollout_info



rollout_infos = []



from pathlib import Path

directory = Path("folders")

for path in directory.iterdir():
    if path.is_file():
        print(path)          # path is already a Path object
        print(type(path))    # <class 'pathlib.PosixPath'> (or WindowsPath)

        resssssss = pkl_utils.load_pkl_zstd(path)

        rollout_infos.append(get_rollout_info(path))

    #get_rollout_info(Path("tradeInfos_snapshot_replay.zstd"))


# ---- SHOULD add multiple rolloutinfos to the list
pkl_utils.save_pkl_zstd(
    rollout_infos,   # must be a list
    Path("rollout_info_25_minutes.zstd")
)




# # ---- load your saved minute lists ----
# replay_minutes = pkl_utils.load_pkl_zstd(Path("my_minutes_replay.zstd"))
# sim_minutes = pkl_utils.load_pkl_zstd(Path("my_minutes_replay.zstd"))


# # make both same length
# n = min(len(replay_minutes), len(sim_minutes))
# print("N IS ")
# print(n)

# # print("replay_minutesreplay_minutesreplay_minutes")
# # for repl in replay_minutes:
# #     print(repl)

# # truncate both to first n

# replay_minutes = replay_minutes[60:99]
# sim_minutes = sim_minutes[60:99]
# """
# print("replay_minutesreplay_minutes")
# print(replay_minutes)
# breakpoint()
# """
# # ---- keep only valid samples (must be exactly 26 minutes) ----
# #if len(replay_minutes) != 26 or len(sim_minutes) != 26:
# #    raise ValueError("Both replay and simulation must contain exactly 26 minutes.")

# # ---- build RolloutInfo object ----
# rollout = RolloutInfo(
#     symbol="AAPL",  # change accordingly
#     start_time=replay_minutes[15].time,  # or sim_minutes[0].time (should match)
#     simulation_minutes=sim_minutes,
#     replay_minutes=replay_minutes,
# )

# # ---- save as list[RolloutInfo] ----
# pkl_utils.save_pkl_zstd(
#     [rollout],   # must be a list
#     Path("rollout_info_25_minutes.zstd")
# )
