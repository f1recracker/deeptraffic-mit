
//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 5;
patchesAhead = 40;
patchesBehind = 10;
trainIterations = 500000;

// the number of other autonomous vehicles controlled by your network
otherAgents = 0; // max of 9

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 0;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
layer_defs.push({
    type: 'fc',
    num_neurons: 5 * (lanesSide * 2 + 1),
    activation: 'relu'
});
layer_defs.push({
    type: 'fc',
    num_neurons: (lanesSide * 2 + 1),
    activation: 'relu'
});
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.9,
    batch_size: 128,
    l1_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 0.05 * trainIterations;
opt.start_learn_threshold = 500;
opt.gamma = 0.9;
opt.learning_steps_total = 0.9 * trainIterations;
opt.learning_steps_burnin = 0.1 * trainIterations;
opt.epsilon_min = 0.2;
opt.epsilon_test_time = 0.1;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
    brain.backward(lastReward);
    var action = brain.forward(state);

    draw_net();
    draw_stats();

    return action;

}

//]]>
