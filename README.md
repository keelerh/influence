# influence

The `influence` library implements the generalized "Influence Model" proposed in Asavathiratham's 2001 PhD thesis.

As a basic example, we define two sites (nodes) in the network: a leader and a follower. The follower meticulously
follows the behavior of the leader. Both sites have two possible statuses, `0`, or `1`, represented by indicator
vectors. We also define a network matrix $D$ and a state-transition matrix $A$ to instantiate the influence model.

```console
>> import numpy as np
>>
>> from influence.influence_model import InfluenceModel
>> from influence.site import Site
>>
>> leader = Site('leader', np.array([[1], [0]]))
>> follower = Site('follower', np.array([[0], [1]]))
>> D = np.array([
>>     [1, 0],
>>     [1, 0],
>> ])
>> A = np.array([
>>     [.5, .5, 1., 0.],
>>     [.5, .5, 0., 1.],
>>     [.5, .5, .5, .5],
>>     [.5, .5, .5, .5],
>> ])
>> model = InfluenceModel([leader, follower], D, A)
>> initial_state = model.get_state_vector()
>> print(initial_state)
```

The initial state of the network is simply a vector stack of the initial statuses of the two sites.

```console
[[1]
 [0]
 [0]
 [1]]
```

Now, we apply the evolution equations of the influence model to progress to the next state of the network.

```console
>> next(model)
>> next_state = model.get_state_vector()
>> print(next_state)
```

We see that the follower has adapted the previous status of the leader.

```console
[[0]
 [1]
 [1]
 [0]]
```

This following behavior continues through subsequent iterations.

```console
>> next(model)
>> next_state = model.get_state_vector()
>> print(next_state)
```

```console
[[1]
 [0]
 [0]
 [1]]
```
