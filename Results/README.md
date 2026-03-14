# Brain MRI Tumor Detection — End-to-End MLOps Pipeline

## Results
### Inference Metrics Monitored

To ensure the **health, performance, and scalability** of the brain MRI inference system, several key metrics are tracked.

#### 1. Inference Request Count
* **What it is:** The total number of inference requests received by the system over time.  
* **Why it matters:**  
  - Shows system usage and demand.  
  - Helps detect spikes in traffic or unusual drops in requests, which could indicate client-side or API issues.

#### 2. Average Inference Latency
* **What it is:** The average time taken to process a single inference request, usually measured in milliseconds.  
* **Why it matters:**  
  - Indicates how fast your model responds under typical load.  
  - Helps identify performance bottlenecks in the model, preprocessing, or API layers.

#### 3. Requests per Second (RPS)
* **What it is:** The number of inference requests handled by the system per second.  
* **Why it matters:**  
  - Measures throughput and scalability of your API.  
  - Ensures the system can handle expected peak loads without performance degradation.

#### 4. P95 Latency (95th Percentile Latency)
* **What it is:** The latency below which 95% of all inference requests fall.  
* **Why it matters:**  
  - Provides a more robust measure than average latency, as it captures tail latency (slow requests).  
  - Helps ensure consistent performance and good user experience even under peak or extreme loads.
#### 5. Overall Inference
* Request count + RPS → tells you how much the system is used and how fast it can serve traffic.
* Average latency + P95 latency → tells you how efficiently and reliably the model is performing.

  ---
  
  ![grafana](Results/imgdump/Pasted image (10).png)
