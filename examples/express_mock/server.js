// Stateful order & export API exercised by the DriftGate validation suites.
// Behavior-identical to the reference server described in docs/ARCHITECTURE.md.
// Only additions over the reference: /healthz readiness endpoint and PORT override.
const express = require("express");

const app = express();
app.use(express.json());

const PORT = Number(process.env.PORT || 3000);
const orders = new Map();
const exportJobs = new Map();

const checkAuth = (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "Unauthorized: Missing or invalid token" });
  }
  next();
};

function getUpdatedOrder(orderId) {
  const order = orders.get(orderId);
  if (!order) return null;

  const elapsedMs = Date.now() - order.createdTimestamp;
  if (order.status !== "CANCELLED") {
    if (elapsedMs >= 15000) order.status = "COMPLETED";
    else if (elapsedMs >= 5000) order.status = "PROCESSING";
  }
  return order;
}

app.get("/healthz", (_req, res) => res.status(200).json({ status: "ok" }));

app.post("/v1/auth/login", (req, res) => {
  const { username, apiKey } = req.body || {};
  if (!username || !apiKey) return res.status(400).json({ error: "Missing credentials" });
  return res.status(200).json({ token: "mock-jwt-bearer-token-12345", expiresIn: 3600 });
});

app.post("/v1/orders", checkAuth, (req, res) => {
  const correlationId = req.headers["x-correlation-id"];
  if (!correlationId) {
    return res.status(400).json({ error: "Missing X-Correlation-ID header" });
  }

  const { customerId, items, shippingAddress } = req.body || {};
  if (!customerId || !Array.isArray(items) || items.length === 0 || !shippingAddress) {
    return res.status(400).json({ error: "Invalid payload" });
  }

  const totalAmount = items.reduce((sum, item) => sum + item.quantity * item.unitPrice, 0);
  const orderId = `ORD-${Math.floor(10000 + Math.random() * 90000)}`;
  const now = new Date();
  const newOrder = {
    orderId,
    customerId,
    items,
    shippingAddress,
    totalAmount,
    status: "PENDING",
    createdAt: now.toISOString(),
    createdTimestamp: now.getTime(),
  };
  orders.set(orderId, newOrder);

  return res.status(202).json({
    orderId,
    status: newOrder.status,
    totalAmount,
    createdAt: newOrder.createdAt,
  });
});

app.get("/v1/orders/:orderId", checkAuth, (req, res) => {
  const order = getUpdatedOrder(req.params.orderId);
  if (!order) return res.status(404).json({ error: "Order not found" });
  return res.status(200).json(order);
});

app.delete("/v1/orders/:orderId", checkAuth, (req, res) => {
  const order = getUpdatedOrder(req.params.orderId);
  if (!order) return res.status(404).json({ error: "Order not found" });
  if (order.status === "COMPLETED") {
    return res.status(409).json({ error: "Cannot cancel completed order" });
  }
  order.status = "CANCELLED";
  orders.set(order.orderId, order);
  return res.status(200).json({ orderId: order.orderId, status: "CANCELLED" });
});

app.post("/v1/exports", checkAuth, (_req, res) => {
  const jobId = `JOB-${Math.floor(10000 + Math.random() * 90000)}`;
  exportJobs.set(jobId, {
    jobId,
    status: "PROCESSING",
    createdTimestamp: Date.now(),
    durationMs: 60000,
  });
  return res.status(202).json({ jobId, status: "PROCESSING", pollIntervalSeconds: 5 });
});

app.get("/v1/exports/:jobId", checkAuth, (req, res) => {
  const job = exportJobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: "Export job not found" });

  if (job.status !== "COMPLETED" && Date.now() - job.createdTimestamp >= job.durationMs) {
    job.status = "COMPLETED";
    job.downloadUrl = `/v1/exports/${job.jobId}/download`;
  }
  return res.status(200).json({
    jobId: job.jobId,
    status: job.status,
    downloadUrl: job.downloadUrl || null,
  });
});

app.get("/v1/exports/:jobId/download", checkAuth, (req, res) => {
  const job = exportJobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: "Export job not found" });
  if (job.status !== "COMPLETED") {
    return res.status(400).json({ error: "Export file is not ready yet" });
  }

  res.setHeader("Content-Type", "text/csv");
  res.setHeader("Content-Disposition", 'attachment; filename="orders_report.csv"');
  return res
    .status(200)
    .send("orderId,status,totalAmount\nORD-10001,COMPLETED,150.00\nORD-10002,CANCELLED,45.50");
});

app.listen(PORT, () => console.log(`Mock API Server running at http://localhost:${PORT}`));
