
# csvname="data/pid_log (3).csv"
# csvname="data/pid_log (4).csv"
# python3 plot_response.py "$csvname" --tmin 0.0 --tmax 1.0

csvs=(
    "data/pid_log_kp=0.05_ki=1_kd=0_lpf_tf=0.01.csv"
    "data/pid_log_kp=0.05_ki=1_kd=0_lpf_tf=0.07.csv"
)

# Optional: comma-separated labels. Keep count == ${#csvs[@]}.
csvlabelstr="one,two"

# python3 plot_compare.py "${csvs[@]}" --labels "$csvlabelstr" --tmin 0.0 --tmax 1.0 --save image/compare.png
python3 plot_compare.py "${csvs[@]}" --tmin 0.0 --tmax 1.0 --save image/compare13.png