close all;

ia = IA(0.01, 1.4);

ia.UseGPU = false;

defaultFilter = @NoFilter;

in_size = 30;
patch_size = 10;

input1 = ia.NewFilterNode(patch_size, defaultFilter);
input2 = ia.NewFilterNode(4, defaultFilter);


tset = 1000;


ta = 1:tset;
noise = 0.05;
line_g = 20;


mm = (ta ./ tset)';
nn = pi * 2;
in = mm * exp(linspace(0,nn,in_size)) - line_g * (1-mm) * linspace(0,nn,in_size) + noise*rand(tset,in_size);

n_patches = tset * floor(in_size/patch_size);

patches = reshape(in, n_patches, patch_size);

in_length = size(in,1);

p_in1 = linspace(0, 2, in_length)';
p_in2 = linspace(0, 2, in_length)';
p_in3 = linspace(-2, 2, in_length)';
p_in4 = linspace(-2, 2, in_length)';

p_in = [p_in1(randsample(1:in_length, in_length)), ...
        p_in2(randsample(1:in_length, in_length)), ...
        p_in3(randsample(1:in_length, in_length)), ...
        p_in4(randsample(1:in_length, in_length))...
        ];

nc = 32;
n_chunk = 2;
im1 = ia.NewIntentionalModule(input1.index, tset, n_chunk, nc); 
% im2 = ia.NewIntentionalModule(input2.index, n_patches, n_chunk, 4); 


% im21 = ia.NewIntentionalModule([im1 im2], tset, n_chunk, nc); 
%     im22 = ia.NewIntentionalModule([im1 im2], tset, 1, 5);
%     im31 = ia.NewIntentionalModule([im21 im22], tset, nc); 
%     im32 = ia.NewIntentionalModule([im21 im22], tset, nc);


%     imf1 = ia.NewIntentionalModule([im1 im2], tset, 1, nc); 
%     imf2 = ia.NewIntentionalModule([im1 im2], tset, 1, nc); 

%     iml = ia.NewIntentionalModule([input1.index input2.index], tset, 1, nc);

outm = ia.NewOutputModule(im1, tset, 2, nc);


s = input1.InputSize();
c = 0.5 * ones(s, 1);
a = -1;
b = 1;



figure(1068);
steps = 5000;
hold on;
erp = plot(1:steps, zeros(1, steps), 'G');
rp = plot(1:steps, zeros(1, steps), 'R');

hold off;
figure(6777);
steps_plot = plot(1:100, zeros(1,100), 'Y');

figure(6789);
in_plot = plot(zeros(1,s));

figure(6790);
hold on;
out_plot1 = plot(zeros(1,steps));
out_plot2 = plot(zeros(1,steps));
%     axis([0 steps -1 1]);
hold off;
x = 1;

out1 = 0;
out2 = 0;

x_old = 0;

prev_rew = 0;

n_in = 1000;
trained_on = 0;

linsp = linspace(0,nn,in_size);


rng('shuffle');
while ia.IsBootstraping()
    out1 = (rand(n_in,1) > 0.5) * 1000;
    out2 = (rand(n_in,1) > 0.5) * 1000;

    
%     in = patches(trained_on+1:min(trained_on + 10, n_patches), :);
%     trained_on = mod(trained_on + 10, n_patches);
    
    mm = rand();
    in = mm .* exp(linsp) - (1-mm) * line_g .* linsp + noise*rand(1,in_size);
    in_patches = reshape(in, in_size / patch_size, patch_size);
    
    
    input1.SetInput(in_patches);
    input2.SetInput([out1, out2, out1 + out2, out1 - out2]);

    ia.Train();

    output = tanh(out1 - out2);


    in_plot.YData = in(1,:);

    out_plot1.YData(1:end-1) = out_plot1.YData(2:end); 
    out_plot1.YData(end) =  out1(1,:);

    out_plot2.YData(1:end-1) = out_plot2.YData(2:end); 
    out_plot2.YData(end) = out2(1,:);
    pause(0.001);
end

linsp = linspace(0,nn,in_size);

output = 0;
pos = rand();
while 1

    pos = pos + 0.005*output;

    mm = pos;

    in = mm .* exp(linsp) - (1-mm) * line_g .* linsp + noise*rand(1,in_size);

    
    in_patches = reshape(in, in_size / patch_size, patch_size);
    
    input1.SetInput(in_patches);
    input2.SetInput([out1, out2, out1 + out2, out1 - out2]);

    rew =  - 0.1*(abs(sum(mean(in, 1))));% - 981

    drew = (rew - prev_rew)/100;
    prev_rew = rew;

%         ia.IncrementReward(drew);
    ia.Update();

    output = ia.GetModuleOutput(outm);

    if isempty(output)
       output = [0,0]; 
    end
    out1 = output(1) * 1000;
    out2 = output(2) * 1000;
    output = (out1 - out2) / 1000;

    if x - x_old > 100

       pos = rand();
       ia.NewEpisode(0);
       x_old = x;
       out1 = 0;
       out2 = 0;
    end

    if rew >= -20

       pos = rand();
       ia.NewEpisode(100);

       steps_plot.YData(1:end-1) = steps_plot.YData(2:end); 
       steps_plot.YData(end) = x - x_old;
       x_old = x;
       out1 = 0;
       out2 = 0;
    end


    in_plot.YData = mean(in, 1);

    rp.YData(1:end-1) = rp.YData(2:end);
    rp.YData(end) = 0.1*(sum(mean(in, 1)));

    erp.YData(1:end-1) = erp.YData(2:end); 
    erp.YData(end) = mean(rp.YData(end-500:end));
%         
    out_plot1.YData(1:end-1) = out_plot1.YData(2:end); 
    out_plot1.YData(end) =  mean([out_plot1.YData(end-1:end), out1]);

    out_plot2.YData(1:end-1) = out_plot2.YData(2:end); 
    out_plot2.YData(end) = mean([out_plot2.YData(end-1:end), out2]);

    pause(0.001);
    x = x+1;
end


