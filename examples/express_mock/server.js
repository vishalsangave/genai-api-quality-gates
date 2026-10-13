// Stateful order & export API exercised by the DriftGate validation suites.
// Behavior-identical to the reference server described in docs/ARCHITECTURE.md.
// Additions over the reference: /healthz readiness endpoint, PORT override, and
// the non-deterministic demo endpoints /v1/recommendations and /v1/insights
// (absent from the reference server).
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

const CATALOG = [
  { sku: "SKU-1001", title: "Thermal Receipt Printer", reason: "Frequently ordered with POS upgrades" },
  { sku: "SKU-1002", title: "Barcode Scanner", reason: "Popular in warehouse expansions" },
  { sku: "SKU-1003", title: "Label Roll Pack", reason: "Consumable paired with scanners" },
  { sku: "SKU-1004", title: "Cash Drawer", reason: "Common add-on for new registers" },
  { sku: "SKU-1005", title: "Card Terminal", reason: "High attach rate with checkout bundles" },
  { sku: "SKU-1006", title: "Cash Drawer", reason: "Standard companion for terminals" },
  { sku: "SKU-1007", title: "Tablet Stand", reason: "Tops accessory list for kiosks" },
  { sku: "SKU-1008", title: "Ethernet Hub", reason: "Frequent companion purchase" },
];

const INSIGHT_TEMPLATES = [
  "Order status updates propagate to every connected operations dashboard within seconds reliably.",
  "Order status updates propagate to every connected operations dashboard within seconds consistently.",
  "Order status updates propagate to every connected operations dashboard within seconds predictably.",
  "Order status updates propagate to every connected operations dashboard within seconds automatically.",
  "Order status updates propagate to every connected operations dashboard within seconds continuously.",
  "Order status updates propagate to every connected operations dashboard within seconds instantaneously.",
];

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

app.get("/v1/recommendations", checkAuth, (req, res) => {
  const raw = req.query.seed;
  let rng = Math.random;
  let seed = null;
  if (raw !== undefined) {
    seed = Number(raw);
    if (!Number.isInteger(seed) || seed < 0) {
      return res.status(400).json({ error: "seed must be a non-negative integer" });
    }
    rng = mulberry32(seed);
  }
  const pool = [...CATALOG];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const recommendations = pool.slice(0, 3).map((item) => ({
    ...item,
    score: Number((0.5 + rng() * 0.49).toFixed(2)),
  }));
  const id = seed === null ? Math.floor(10000 + Math.random() * 90000) : (seed % 90000) + 10000;
  return res.status(200).json({
    requestId: `REQ-${id}`,
    generatedAt: new Date().toISOString(),
    recommendations,
  });
});

app.post("/v1/insights", checkAuth, (req, res) => {
  const { customerId, prompt } = req.body || {};
  if (!customerId || !prompt) return res.status(400).json({ error: "Invalid payload" });
  const insight = INSIGHT_TEMPLATES[Math.floor(Math.random() * INSIGHT_TEMPLATES.length)];
  const confidence = Number((0.5 + Math.random() * 0.49).toFixed(2));
  return res.status(200).json({ insight, confidence });
});

app.listen(PORT, () => console.log(`Mock API Server running at http://localhost:${PORT}`));
