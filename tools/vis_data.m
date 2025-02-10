%% Data Import
clear; clc;

data_folder = '/home/wcompton/repos/IsaacLab/outputs/planner_tracker/';
fn = '2025-02-09/17-58-36/epoch_0.mat';
load([data_folder fn]);
legs = {'FL', 'FR', 'RL', 'RR'};

cmd = obs(:, :, 7:9);
q = obs(:, :, 10: 10 + 11);
dq = obs(:, :, 10 + 12: 10 + 12 + 11);

%% Now plot
inds = [100, 500];
robot = 1;
act_plt = squeeze(act(robot, :, :));
q_plt = squeeze(q(robot, :, :));
figure(1)
clf
for i = 1:4
    subplot(2, 2, i)
    l = 3 * (i - 1) + 1;
    u = 3 * i;
    hold on
    plot(act_plt(:, i:4:end), '--')
    set(gca, "ColorOrderIndex", 1)
    plot(q_plt(:, i:4:end))
    legend('Hip des', 'Thigh des', 'Calf des', 'Hip', 'Thigh', 'Calf')
    xlabel('Time')
    ylabel('Joint Position')
    title(['RL ' legs{i}])
    xlim(inds);
end

%% Mujoco

tbl = readtable('/home/wcompton/repos/go2-navigation/rl_vel_tracking_data.csv');

inds_mjc = [200, 300];

cmd_mjc = tbl{:, 7:9};
q_mjc = tbl{:, 10: 10 + 11};
dq_mjc = tbl{:, 10 + 12: 10 + 12 + 11};
act_mjc = tbl{:, 10+24:10+24+11};

figure(2)
clf
for i = 1:4
    subplot(2, 2, i)
    l = 3 * (i - 1) + 1;
    u = 3 * i;
    hold on
    plot(act_mjc(:, i:4:end), '--')
    set(gca, "ColorOrderIndex", 1)
    plot(q_mjc(:, i:4:end))
    legend('Hip des', 'Thigh des', 'Calf des', 'Hip', 'Thigh', 'Calf')
    xlabel('Time')
    ylabel('Joint Position')
    title(['MJC ' legs{i}])
    % xlim(inds_mjc);
end

figure(3)
plot(cmd)
xlim(inds_mjc)

%%
load('joint_traj.mat')

figure(4)
clf


plot(joint_traj, '--')
legend('Hip des', 'Thigh des', 'Calf des')
xlabel('Time')
ylabel('Joint Position')
