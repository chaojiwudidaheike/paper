Our data generation process will be introduced in three parts, first introducing the data parameter description, then introducing the

real-world data, and then introducing the random data.

### 1\. Dataset Parameter Descriptions

###### UAV：

* &nbsp;	uRge: Instantaneous flight speed of the UAV, randomly drawn from the interval 0.8 km/min to 0.9 km/min under normal operating conditions.
* &nbsp;	uPow: Remaining battery capacity at the time the UAV comes online; initial battery level is randomly selected to correspond to a full-charge range of 20 km to 40 km.
* &nbsp;	FullPower: Maximum battery capacity when fully charged, also corresponding to a full-charge range randomly between 20 km and 40 km; must satisfy FullPower > uPow.
* &nbsp;	uLoc: Geographic coordinates of the UAV’s initial deployment location.
* &nbsp;	U\_uptime: Timestamp at which the UAV is activated (goes online).
* &nbsp;	U\_downtime: Timestamp at which the UAV is deactivated (goes offline).

###### Worker：

* &nbsp;	wRge: Walking speed of the worker, randomly drawn from the interval 0.2 km/min to 0.3 km/min under normal adult locomotion.
* &nbsp;	wLoc: Geographic coordinates of the worker’s initial position.
* &nbsp;	W\_uptime: Timestamp at which the worker begins operation (goes online).
* &nbsp;	W\_downtime: Timestamp at which the worker ends operation (goes offline).

###### Vehicle：

* &nbsp;	cRge: Driving speed of the vehicle, randomly drawn from the interval 0.5 km/min to 0.6 km/min under normal traffic conditions.
* &nbsp;	cLoc: Geographic coordinates of the vehicle’s initial position.
* &nbsp;	C\_uptime: Timestamp at which the vehicle enters service (goes online).
* &nbsp;	C\_downtime: Timestamp at which the vehicle leaves service (goes offline).
* &nbsp;	C\_chargePow: Charging power of the vehicle’s battery (energy replenished per minute); used in conjunction with the UAV’s remaining battery capacity to compute recharge duration.

###### Task Point：

* &nbsp;	tLoc: Geographic coordinates of a task location.
* &nbsp;	T\_costime: task completion time required to complete the task; 

###### Charge Point：

* &nbsp;	chLoc: Geographic coordinates of a charge station.

### 2\. Real-world dataset：

###### (1) Bicycle Riding Order Data (BicycleOrder)

* Orders with start times between 08:00 and 12:00 and total riding durations exceeding 70 minutes were treated as agents.
* The study area was discretized into 1 km × 1 km grid cells, yielding a 25 × 16 grid.
* Agents were categorized by their mean straight-line speed:

&nbsp;	- > 1.5 km/h → UAV

&nbsp;	- 1.0–1.5 km/h → Vehicle

&nbsp;	- < 1.0 km/h → Worker

* For each order, the origin coordinate was recorded as the agent’s initial position; the order’s start and end times were logged as the agent’s online and offline times, respectively.
* All public bicycle stations were designated as task points (N = 129), and an additional set of 30 non-overlapping recharge points was generated at random within the region.

###### (2) Taxi Trajectory Data (TaxiTrajectories)

* Trips beginning between 08:00 and 12:00 with continuous movement durations exceeding 90 minutes were treated as agents.
* The same 1 km × 1 km discretization produced a 30 × 30 grid.
* Agents were classified by mean straight-line speed:

&nbsp;	- > 32 km/h → UAV

&nbsp;	- 25–32 km/h → Vehicle

&nbsp;	- < 25 km/h → Worker

* Each trip’s origin was recorded as the agent’s initial position, with start and end timestamps marking the agent’s online and offline times.
* Every trip origin location was designated as a task point (N = 162), and 60 non-overlapping recharge points were randomly generated within the same area.

###### (3) Didi Ride-Hailing Order Data (DidiOrders)

* Orders initiated between 08:00 and 12:00 with durations exceeding 50 minutes were treated as agents.
* A 10 × 9 grid of 1 km × 1 km cells was used to discretize the service area.
* Agents were categorized by mean straight-line speed:

&nbsp;	- > 7 km/h → UAV

&nbsp;	- 5–7 km/h → Vehicle

&nbsp;	- < 5 km/h → Worker

* The pickup location of each order was designated as the agent’s initial position, and the order’s start and end times defined the agent’s online and offline times.
* Grid cells containing four or more orders were marked as task points (N = 44), and 10 non-overlapping recharge points were randomly generated within the region.

### 3\. Random Dataset Description

###### Fixed Parameters

Map size:

* Discretized into 1 km × 1 km grid cells, forming a (Map\_size × Map\_size) grid.
* Experiment time limit: LIMIT\_TIME
* Agent online duration: online\_time
* Population counts:

&nbsp;	Number of UAVs: uav\_num

&nbsp;	Number of workers: worker\_num

&nbsp;	Number of vehicles: car\_num

&nbsp;	Number of tasks: task\_num

&nbsp;	Number of recharge points: charge\_num

###### Random Variables

* Agent online time (X\_uptime):

&nbsp;	Uniformly sampled in \[0, LIMIT\_TIME – online\_time].

* Agent offline time (X\_downtime):

&nbsp;	Defined as X\_uptime + online\_time.

* Agent initial coordinates (x, y):

&nbsp;	Independently and uniformly sampled in \[0, Map\_size – 1], with repetition allowed.

* Task-point and recharge-point coordinates (x, y):

&nbsp;	Independently and uniformly sampled in \[0, Map\_size – 1], without repetition.

* UAV full battery capacity (Fullpower):

&nbsp;	Uniformly sampled in \[20, 40].

* UAV initial battery level (uPow):

&nbsp;	Uniformly sampled in \[0, Fullpower].

* UAV speed (uRge):

&nbsp;	Uniformly sampled in \[0.8, 0.9].

* Worker speed (wRge):

&nbsp;	Uniformly sampled in \[0.2, 0.3].

* Vehicle speed (cRge):

&nbsp;	Uniformly sampled in \[0.5, 0.6].

