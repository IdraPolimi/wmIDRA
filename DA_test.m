close all;

n_features = 1;

n_steps = 1000;
max_time = 50;

steps = linspace(0, max_time, n_steps);

x = zeros(n_features, n_steps);
reward = zeros(1, n_steps);

w = 0.003*rand(n_features, n_steps);
V = zeros(1, n_steps);
Vtot = zeros(1, n_steps);
delta = zeros(1, n_steps);


mu = 0.4;

reward(:) = 2*exp(-1 * (steps - max_time/1.1).^2);

hold on;
pr = plot(steps, reward, '--b');
px1 = plot(steps, x(1,:), ':g');
pd = plot(steps, delta, 'k');
axis([0 max_time -1.5 2]);
legend('Reward', 'Sensorial Cue', 'Value Function', 'TD Error');
hold off;



n_episodes = 1400000;
g = 1;

for episode = 1:n_episodes
    
    x(1,:) = max(2*exp(-g * (steps - max_time/3).^2) + 2*exp(-g * (steps - max_time/2).^2),0.2 * (steps > 16));

    
    if episode > 50000
        reward(:) = zeros(1,n_steps);
    end
    episode
    
    for t = 2:n_steps-1
        
        V(t) = sum( x(:, t) .* w(:, t) );
        Vdot = V(t) - V(t-1);
        delta(t) = reward(t-1) + V(t) - V(t-1);
        
        w(:, t-1) = w(:, t-1) + mu * x(:,t-1) * delta(t);
    end
    
    
    % update plots
    pr.YData = tanh(reward);
    pd.YData = tanh(delta);
    px1.YData = tanh(x(1,:));
    pause(0.001);
end