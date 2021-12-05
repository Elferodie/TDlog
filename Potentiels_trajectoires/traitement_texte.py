def combine_texts(texts, new_name="texts.txt"):
    with open(new_name, "w") as combined_text:
        for text in texts:
            with open(text) as t:
                combined_text.write(t.read())
    return combined_text


def select_trajectory(trajectory, save=1000, new_name="light_trajectory.txt"):
    with open(new_name, "w") as new_traj:
        with open(trajectory) as traj:
            lines = traj.readlines()
            for i in range(len(lines)):
                if i % save == 0:
                    new_traj.write(lines[i])
    return new_traj

#combine_texts(["long_trajectory00.txt", "long_trajectory01.txt", "long_trajectory02.txt", "long_trajectory03.txt"], new_name="complete_trajectory.txt")
#select_trajectory("complete_trajectory.txt", save=100, new_name=f"complete_trajectory_100.txt")
#select_trajectory("complete_trajectory.txt", save=1000, new_name=f"complete_trajectory_1000.txt")

