# Deployment

The service is packaged as a container image and deployed to Kubernetes. A
Deployment manages the running pods, and a horizontal pod autoscaler adds or
removes replicas based on CPU usage and request latency.

Configuration is supplied through environment variables and mounted secrets.
Rolling updates replace pods gradually so there is no downtime during a
release. If a new version fails its readiness probe, the rollout is halted
automatically.
