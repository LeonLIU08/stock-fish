/* 趋势结构图。只读 view model，不重新拟合。 */
const TrendChart = (function () {
  function mount(container, viewModel) {
    if (!container) {
      throw new Error("缺少图表容器");
    }
    if (typeof echarts === "undefined") {
      throw new Error("缺少 echarts");
    }
    const closes = (viewModel && viewModel.closes) || [];
    if (!closes.length) {
      container.textContent = "没有可绘制的收盘价。";
      return { reset: function () {}, destroy: function () {}, focus: function () { return false; }, setScales: function () {}, setExtra: function () {}, state: function () { return {}; }, onChange: function () {} };
    }

    const priceAxis = viewModel.price_axis === "log" ? "log" : "value";
    const labels = viewModel.labels || {};
    const colors = viewModel.colors || {};
    const styles = viewModel.styles || {};
    const barByIndex = new Map();
    closes.forEach(function (bar) { barByIndex.set(bar.index, bar); });
    const pivotsByIndex = new Map();
    (viewModel.pivots || []).forEach(function (pivot) {
      const bucket = pivotsByIndex.get(pivot.index) || [];
      bucket.push(pivot);
      pivotsByIndex.set(pivot.index, bucket);
    });

    const legendScale = new Map();
    const wantedNames = [];
    function track(name, scale) {
      if (!legendScale.has(name)) {
        legendScale.set(name, scale);
        wantedNames.push(name);
      }
    }

    const series = [];
    const focusIndex = { boundary: new Map(), zone: new Map(), pivot: new Map(), segment: new Map() };

    series.push({
      id: "closes",
      name: "收盘价",
      type: "line",
      data: closes.map(function (bar) { return [bar.index, bar.close]; }),
      showSymbol: false,
      symbol: "none",
      lineStyle: { color: colors.close || "#E6E1D6", width: 1.4 },
      emphasis: { disabled: true },
      z: 2,
      markLine: viewModel.as_of_index == null ? undefined : {
        symbol: ["none", "none"],
        label: { formatter: "as-of", color: colors.as_of || "#E0A45A" },
        lineStyle: { color: colors.as_of || "#E0A45A", type: "dashed", width: 1 },
        data: [{ xAxis: viewModel.as_of_index }],
      },
    });
    track("收盘价", null);

    const pivotsByScale = groupBy((viewModel.pivots || []), function (item) { return item.scale; });
    Object.keys(pivotsByScale).forEach(function (scale) {
      const name = scaleName(scale) + " 拐点";
      track(name, scale);
      series.push({
        id: "pivots-" + scale,
        name: name,
        type: "scatter",
        data: pivotsByScale[scale].map(function (pivot) {
          focusIndex.pivot.set(pivot.id, pivot);
          return Object.assign({ value: [pivot.index, pivot.price] }, pivot);
        }),
        symbol: function (_value, params) { return params.data.confirmed ? "circle" : "emptyCircle"; },
        symbolSize: 8,
        itemStyle: {
          color: function (params) {
            return params.data.confirmed ? roleColor(params.data.role) : "rgba(0,0,0,0)";
          },
          borderColor: function (params) { return roleColor(params.data.role); },
          borderWidth: 1.6,
        },
        z: 4,
      });
    });

    (viewModel.boundaries || []).forEach(function (line) {
      focusIndex.boundary.set(line.id, line);
      const expired = line.layer === "expired";
      const name = expired ? "已过期" : scaleName(line.scale) + " 趋势线";
      track(name, expired ? null : line.scale);
      series.push(lineSeries(line.id, name, line.solid, lineStyle(line, expired), line));
      if (line.projection && line.projection.length === 2) {
        const projectionName = expired ? "已过期" : scaleName(line.scale) + " 投影";
        track(projectionName, expired ? null : line.scale);
        series.push(lineSeries(line.id + "|projection", projectionName, line.projection, {
          color: roleColor(line.role),
          width: 1.2,
          type: "dashed",
          opacity: 0.85,
        }, line));
      }
    });

    const zonesByName = {};
    (viewModel.zones || []).forEach(function (zone) {
      focusIndex.zone.set(zone.id, zone);
      const name = scaleName(zone.scale) + " " + zoneTypeName(zone.label);
      track(name, zone.scale);
      if (!zonesByName[name]) zonesByName[name] = [];
      const lower = zone.lower;
      const upper = zone.upper;
      const ended = zone.status === "broken" || zone.status === "expired";
      zonesByName[name].push({
        value: [lower[0][0], lower[1][0], lower[0][1], lower[1][1], upper[1][1], upper[0][1]],
        objectId: zone.id,
        scale: zone.scale,
        label: zone.label,
        status: zone.status,
        itemStyle: {
          color: colors[zone.label] || "rgba(126,182,255,0.16)",
          borderColor: colors[zone.label + "_stroke"] || "rgba(126,182,255,0.8)",
          borderWidth: ended ? 1.4 : 1,
          borderType: ended ? "dashed" : "solid",
        },
      });
    });
    Object.keys(zonesByName).forEach(function (name) {
      const sample = zonesByName[name][0];
      series.push({
        id: "zones-" + name,
        name: name,
        type: "custom",
        coordinateSystem: "cartesian2d",
        renderItem: zoneRenderItem,
        data: zonesByName[name],
        clip: true,
        z: 1,
        scaleKey: sample.scale,
        itemStyle: { color: colors[sample.label + "_stroke"] || colors[sample.label] || "#7EB6FF" },
      });
    });

    const breakoutsByScale = {};
    (viewModel.boundaries || []).concat(viewModel.zones || []).forEach(function (item) {
      if (!item.breakout || item.breakout.price == null || item.layer === "expired") return;
      const scale = item.scale;
      if (!breakoutsByScale[scale]) breakoutsByScale[scale] = [];
      breakoutsByScale[scale].push({
        value: [item.breakout.index, item.breakout.price],
        objectId: item.id,
        scale: scale,
        role: item.role || item.label,
        status: "broken",
        priceCrossIndex: item.breakout.price_cross_index,
      });
    });
    Object.keys(breakoutsByScale).forEach(function (scale) {
      const name = scaleName(scale) + " 突破";
      track(name, scale);
      series.push({
        id: "break-" + scale,
        name: name,
        type: "scatter",
        data: breakoutsByScale[scale],
        symbol: "diamond",
        symbolSize: 12,
        itemStyle: { color: colors.breakout || "#FF5C7A" },
        z: 5,
      });
    });

    const segmentData = [];
    (viewModel.segments || []).forEach(function (segment, index) {
      focusIndex.segment.set(segment.id, segment);
      if (index) segmentData.push(null);
      segment.points.forEach(function (pair) {
        segmentData.push({
          value: pair,
          objectId: segment.id,
          scale: segment.scale,
          direction: segment.direction,
        });
      });
    });
    if (segmentData.length) {
      track("基础线段", null);
      series.push({
        id: "segments",
        name: "基础线段",
        type: "line",
        data: segmentData,
        showSymbol: false,
        symbol: "none",
        connectNulls: false,
        lineStyle: { color: colors.segment || "rgba(230,225,214,0.45)", width: 1 },
        emphasis: { disabled: true },
        z: 1,
      });
    }

    series.push({
      id: "focus-line",
      name: "选中",
      type: "line",
      data: [],
      showSymbol: false,
      symbol: "none",
      connectNulls: false,
      lineStyle: { color: colors.focus || "#F3D58A", width: 3.4 },
      z: 6,
      silent: true,
    });
    series.push({
      id: "focus-point",
      name: "选中",
      type: "scatter",
      data: [],
      symbolSize: 12,
      itemStyle: { color: colors.focus || "#F3D58A" },
      z: 7,
      silent: true,
    });

    const scaleSet = new Set(viewModel.default_scales || ["mid"]);
    let projectionOn = true;
    const extras = { "基础线段": false, "已过期": false };
    const selected = {};
    applySelection(selected);

    const priceBounds = paddedPriceBounds(viewModel, priceAxis === "log");
    const chart = echarts.init(container, null, { renderer: "canvas" });
    chart.setOption({
      backgroundColor: "transparent",
      animationDurationUpdate: 0,
      textStyle: { fontFamily: '"Avenir Next", "PingFang SC", "Noto Sans SC", sans-serif' },
      graphic: [{
        type: "text",
        left: 8,
        top: 6,
        style: {
          text: viewModel.axis_label || "",
          fill: "#a39c90",
          font: '12px "Avenir Next", "PingFang SC", sans-serif',
        },
      }],
      grid: { left: 8, right: 36, top: 28, bottom: 112, containLabel: true },
      legend: {
        type: "scroll",
        bottom: 40,
        data: wantedNames,
        selected: selected,
        textStyle: { color: "#c8c2b4" },
        pageIconColor: "#c8c2b4",
        pageTextStyle: { color: "#c8c2b4" },
      },
      tooltip: {
        trigger: "axis",
        confine: true,
        backgroundColor: "#262017",
        borderColor: "#4a4338",
        textStyle: { color: "#f3efe4", fontSize: 12 },
        axisPointer: { type: "cross", lineStyle: { color: "rgba(224,164,90,0.45)" } },
        formatter: tooltipFormatter,
      },
      xAxis: {
        type: "value",
        axisLine: { lineStyle: { color: "#3a342c" } },
        axisTick: { show: false },
        axisLabel: { color: "#a39c90", hideOverlap: true, margin: 12, formatter: tickText },
        splitLine: { show: false },
      },
      yAxis: {
        type: priceAxis,
        min: priceBounds.min,
        max: priceBounds.max,
        scale: false,
        splitNumber: 5,
        logBase: 10,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: "#a39c90", margin: 10, formatter: formatPrice },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
      },
      dataZoom: [
        { id: "x-pan", type: "inside", xAxisIndex: 0, filterMode: "none", zoomOnMouseWheel: false, moveOnMouseMove: false, moveOnMouseWheel: false },
        {
          id: "x-slider",
          type: "slider",
          xAxisIndex: 0,
          filterMode: "none",
          bottom: 10,
          height: 18,
          borderColor: "#3a342c",
          backgroundColor: "#1a1814",
          fillerColor: "rgba(224,164,90,0.22)",
          handleStyle: { color: "#e0a45a", borderColor: "#e0a45a" },
          textStyle: { color: "#a39c90" },
          brushSelect: false,
          labelFormatter: tickText,
        },
      ],
      series: series,
    });
    chart.resize();

    const initial = extents();
    const fullX = { min: initial.x[0], max: initial.x[1] };
    const fullY = { min: priceBounds.min, max: priceBounds.max };
    let listener = null;
    let drag = null;
    const cleanups = [];

    function on(target, type, handler, options) {
      target.addEventListener(type, handler, options);
      cleanups.push(function () { target.removeEventListener(type, handler, options); });
    }

    on(container, "wheel", onWheel, { passive: false });
    on(container, "pointerdown", onPointerDown);
    on(window, "pointermove", onPointerMove);
    on(window, "pointerup", onPointerUp);
    on(container, "dblclick", onDoubleClick);
    on(window, "resize", function () { chart.resize(); });
    chart.on("legendselectchanged", onLegend);
    chart.on("click", function (params) {
      const objectId = params.data && (params.data.objectId || params.data.id);
      if (objectId) focus(objectId);
    });
    container.style.cursor = "grab";
    container.style.touchAction = "none";

    function reset() {
      dispatchX(fullX.min, fullX.max);
      dispatchY(fullY.min, fullY.max);
      chart.setOption({ series: [{ id: "focus-line", data: [] }, { id: "focus-point", data: [] }] });
    }

    function destroy() {
      cleanups.forEach(function (off) { off(); });
      chart.dispose();
    }

    function focus(id) {
      const boundary = focusIndex.boundary.get(id);
      const zone = focusIndex.zone.get(id);
      const pivot = focusIndex.pivot.get(id);
      const segment = focusIndex.segment.get(id);
      if (!boundary && !zone && !pivot && !segment) return false;
      if (pivot) {
        zoomTo(pivot.index, pivot.index);
        chart.setOption({
          series: [
            { id: "focus-line", data: [] },
            { id: "focus-point", data: [{ value: [pivot.index, pivot.price] }] },
          ],
        });
        return true;
      }
      if (zone) {
        zoomTo(zone.lower[0][0], zone.lower[1][0]);
        chart.setOption({
          series: [
            {
              id: "focus-line",
              data: [
                { value: zone.lower[0] },
                { value: zone.lower[1] },
                null,
                { value: zone.upper[0] },
                { value: zone.upper[1] },
              ],
            },
            { id: "focus-point", data: [] },
          ],
        });
        return true;
      }
      if (segment) {
        zoomTo(segment.points[0][0], segment.points[1][0]);
        chart.setOption({
          series: [
            { id: "focus-line", data: segment.points.map(function (pair) { return { value: pair }; }) },
            { id: "focus-point", data: [] },
          ],
        });
        return true;
      }
      const end = boundary.projection ? boundary.projection[1][0] : boundary.solid[1][0];
      zoomTo(boundary.solid[0][0], end);
      chart.setOption({
        series: [
          { id: "focus-line", data: boundary.solid.map(function (pair) { return { value: pair }; }) },
          { id: "focus-point", data: [] },
        ],
      });
      return true;
    }

    function setScales(names) {
      scaleSet.clear();
      (names || []).forEach(function (name) { scaleSet.add(name); });
      const next = {};
      applySelection(next);
      chart.setOption({ legend: { selected: next } });
      emit();
    }

    function setExtra(name, on) {
      if (name === "投影") projectionOn = !!on;
      else extras[name] = !!on;
      const next = {};
      applySelection(next);
      chart.setOption({ legend: { selected: next } });
      emit();
    }

    function state() {
      return {
        scales: Array.from(scaleSet),
        projection: projectionOn,
        segments: !!extras["基础线段"],
        expired: !!extras["已过期"],
      };
    }

    function onChange(fn) { listener = fn; }

    function emit() { if (listener) listener(state()); }

    function applySelection(target) {
      wantedNames.forEach(function (name) {
        const scale = legendScale.get(name);
        if (name === "基础线段" || name === "已过期") target[name] = !!extras[name];
        else if (scale == null) target[name] = true;
        else if (name.indexOf("投影") >= 0) target[name] = scaleSet.has(scale) && projectionOn;
        else target[name] = scaleSet.has(scale);
      });
    }

    function onLegend(event) {
      const chosen = event.selected || {};
      const next = new Set();
      legendScale.forEach(function (scale, name) {
        if (scale && chosen[name] && name.indexOf("趋势线") >= 0) next.add(scale);
      });
      if (next.size) {
        scaleSet.clear();
        next.forEach(function (scale) { scaleSet.add(scale); });
      }
      const projectionNames = wantedNames.filter(function (name) { return name.indexOf("投影") >= 0; });
      if (projectionNames.length) {
        projectionOn = projectionNames.some(function (name) { return chosen[name]; });
      }
      extras["基础线段"] = !!chosen["基础线段"];
      extras["已过期"] = !!chosen["已过期"];
      emit();
    }

    function onWheel(event) {
      if (!inGrid(event)) return;
      event.preventDefault();
      const origin = pixelToData(event);
      if (!origin) return;
      const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
      zoomAround(origin[0], origin[1], factor);
    }

    function onDoubleClick(event) {
      if (!inGrid(event)) return;
      reset();
    }

    function onPointerDown(event) {
      if (event.button !== 0 || !inGrid(event)) return;
      const ext = extents();
      const px0 = chart.convertToPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [ext.x[0], ext.y[0]]);
      const px1 = chart.convertToPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [ext.x[1], ext.y[1]]);
      if (!px0 || !px1 || px0[0] === px1[0] || px0[1] === px1[1]) return;
      drag = { clientX: event.clientX, clientY: event.clientY, ext: ext, px0: px0, px1: px1, pointerId: event.pointerId };
      container.setPointerCapture(event.pointerId);
      container.style.cursor = "grabbing";
    }

    function onPointerMove(event) {
      if (!drag || event.pointerId !== drag.pointerId) return;
      event.preventDefault();
      const dx = event.clientX - drag.clientX;
      const dy = event.clientY - drag.clientY;
      const xPerPx = (drag.ext.x[1] - drag.ext.x[0]) / (drag.px1[0] - drag.px0[0]);
      shiftLinear(drag.ext.x[0], drag.ext.x[1], -dx * xPerPx, fullX, dispatchX);
      if (priceAxis === "log" && drag.ext.y[0] > 0 && drag.ext.y[1] > 0 && fullY.min > 0 && fullY.max > 0) {
        const a = Math.log(drag.ext.y[0]);
        const b = Math.log(drag.ext.y[1]);
        const per = (b - a) / (drag.px1[1] - drag.px0[1]);
        shiftLog(a, b, -dy * per, Math.log(fullY.min), Math.log(fullY.max), function (min, max) {
          dispatchY(Math.exp(min), Math.exp(max));
        });
      } else {
        const yPerPx = (drag.ext.y[1] - drag.ext.y[0]) / (drag.px1[1] - drag.px0[1]);
        shiftLinear(drag.ext.y[0], drag.ext.y[1], -dy * yPerPx, fullY, dispatchY);
      }
    }

    function onPointerUp(event) {
      if (!drag || event.pointerId !== drag.pointerId) return;
      drag = null;
      container.style.cursor = "grab";
    }

    function zoomAround(originX, originY, factor) {
      const ext = extents();
      zoomLinear(ext.x[0], ext.x[1], originX, factor, fullX, Math.min(4, fullX.max - fullX.min), dispatchX);
      if (priceAxis === "log" && originY > 0 && ext.y[0] > 0 && ext.y[1] > 0 && fullY.min > 0) {
        const a = Math.log(ext.y[0]);
        const b = Math.log(ext.y[1]);
        const fullA = Math.log(fullY.min);
        const fullB = Math.log(fullY.max);
        const minSpan = Math.min(Math.abs(fullB - fullA) * 0.04, Math.abs(fullB - fullA));
        zoomLinear(a, b, Math.log(originY), factor, { min: fullA, max: fullB }, minSpan, function (min, max) {
          dispatchY(Math.exp(min), Math.exp(max));
        });
      } else {
        const minSpan = Math.min((fullY.max - fullY.min) * 0.04, fullY.max - fullY.min);
        zoomLinear(ext.y[0], ext.y[1], originY, factor, fullY, minSpan, dispatchY);
      }
    }

    function zoomTo(start, end) {
      const span = Math.max(end - start, 1);
      const pad = Math.max(4, span * 0.45);
      let left = start - pad;
      let right = end + pad;
      if (right - left < 12 && fullX.max - fullX.min >= 12) {
        const mid = (start + end) / 2;
        left = mid - 6;
        right = mid + 6;
      }
      left = Math.max(fullX.min, left);
      right = Math.min(fullX.max, right);
      dispatchX(left, right);
    }

    function dispatchX(min, max) {
      chart.dispatchAction({ type: "dataZoom", dataZoomIndex: 0, startValue: min, endValue: max });
      chart.dispatchAction({ type: "dataZoom", dataZoomIndex: 1, startValue: min, endValue: max });
    }

    function dispatchY(min, max) {
      chart.setOption({ yAxis: { min: min, max: max } });
    }

    function extents() {
      const grid = chart.getModel().getComponent("grid", 0).coordinateSystem.getRect();
      const low = chart.convertFromPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [grid.x, grid.y + grid.height]);
      const high = chart.convertFromPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [grid.x + grid.width, grid.y]);
      if (low && high && isFinite(low[0]) && isFinite(low[1]) && isFinite(high[0]) && isFinite(high[1])) {
        return {
          x: [Math.min(low[0], high[0]), Math.max(low[0], high[0])],
          y: [Math.min(low[1], high[1]), Math.max(low[1], high[1])],
        };
      }
      return {
        x: chart.getModel().getComponent("xAxis", 0).axis.scale.getExtent(),
        y: chart.getModel().getComponent("yAxis", 0).axis.scale.getExtent(),
      };
    }

    function inGrid(event) {
      const rect = container.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const grid = chart.getModel().getComponent("grid", 0).coordinateSystem.getRect();
      return x >= grid.x && x <= grid.x + grid.width && y >= grid.y && y <= grid.y + grid.height;
    }

    function pixelToData(event) {
      const rect = container.getBoundingClientRect();
      const data = chart.convertFromPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [event.clientX - rect.left, event.clientY - rect.top]);
      if (!data || !isFinite(data[0]) || !isFinite(data[1])) return null;
      return data;
    }

    function tooltipFormatter(params) {
      const list = Array.isArray(params) ? params : [params];
      let index = null;
      list.forEach(function (item) {
        if (index != null) return;
        const value = item.value || (item.data && item.data.value);
        if (value && isFinite(value[0])) index = Math.round(value[0]);
      });
      if (index == null) return "";
      const bar = barByIndex.get(index);
      const lines = [];
      lines.push(esc(bar ? bar.label : "序号 " + index) + " · 序号 " + index);
      if (bar) lines.push("收盘价 " + formatPrice(bar.close));
      (pivotsByIndex.get(index) || []).forEach(function (pivot) {
        const role = (labels.roles && labels.roles[pivot.role]) || pivot.role;
        const scale = scaleName(pivot.scale);
        const stateText = pivot.confirmed ? "已确认" : "临时，尚未确认";
        let text = scale + " " + role + " · " + stateText;
        text += "<br>极值时间 " + esc(pivot.extreme_time || "—");
        if (pivot.confirmed) text += "<br>确认时间 " + esc(pivot.confirm_time || "—");
        if (pivot.threshold != null) text += "<br>确认门槛 " + formatNum(pivot.threshold);
        lines.push(text);
      });
      let extra = 0;
      (viewModel.boundaries || []).forEach(function (line) {
        if (!covers(line, index)) return;
        if (lines.length > 8) { extra += 1; return; }
        lines.push(structureText(line, "趋势线"));
      });
      (viewModel.zones || []).forEach(function (zone) {
        const start = zone.lower[0][0];
        const end = zone.lower[1][0];
        if (index < start || index > end) return;
        if (lines.length > 8) { extra += 1; return; }
        const label = (labels.zones && labels.zones[zone.label]) || zone.label;
        lines.push(scaleName(zone.scale) + " " + label + " · " + statusText(zone.status));
      });
      if (extra) lines.push("另有 " + extra + " 条结构盖住这一根");
      return lines.join("<br>");
    }

    function structureText(line, kind) {
      const role = (labels.roles && labels.roles[line.role]) || line.role || kind;
      let text = scaleName(line.scale) + " " + role + " · " + statusText(line.status);
      if (line.normalized_slope != null) text += "<br>归一化斜率 " + formatNum(line.normalized_slope);
      return text;
    }

    function covers(line, index) {
      if (!line.solid || line.solid.length < 2) return false;
      const start = line.solid[0][0];
      const solidEnd = line.solid[1][0];
      const far = line.projection ? line.projection[1][0] : solidEnd;
      return index >= start && index <= far;
    }

    function scaleName(scale) {
      return (labels.scales && labels.scales[scale]) || scale || "";
    }

    function zoneTypeName(label) {
      return (labels.zones && labels.zones[label]) || label || "区间";
    }

    function statusText(status) {
      return (labels.statuses && labels.statuses[status]) || status || "";
    }

    function roleColor(role) {
      return colors[role] || colors.pending || "#B7B1A6";
    }

    function lineStyle(line, expired) {
      const style = styles[line.scale] || { width: 1.5, type: "solid" };
      return {
        color: roleColor(line.role),
        width: style.width,
        type: expired ? "dotted" : style.type,
        opacity: 1,
      };
    }

    function lineSeries(id, name, points, style, extra) {
      return {
        id: id,
        name: name,
        type: "line",
        data: (points || []).map(function (pair) {
          return {
            value: pair,
            objectId: extra.id,
            scale: extra.scale,
            role: extra.role,
            status: extra.status,
          };
        }),
        showSymbol: false,
        symbol: "none",
        lineStyle: {
          color: style.color,
          width: style.width,
          type: style.type,
          opacity: style.opacity == null ? 1 : style.opacity,
        },
        itemStyle: { color: style.color },
        emphasis: { disabled: true },
        z: 3,
      };
    }

    function tickText(value) {
      const index = Math.round(value);
      if (Math.abs(value - index) > 0.15) return "";
      const bar = barByIndex.get(index);
      return bar ? bar.label : "";
    }

    return { reset: reset, destroy: destroy, focus: focus, setScales: setScales, setExtra: setExtra, state: state, onChange: onChange };
  }

  function zoneRenderItem(params, api) {
    const start = api.value(0);
    const end = api.value(1);
    const points = [
      api.coord([start, api.value(2)]),
      api.coord([end, api.value(3)]),
      api.coord([end, api.value(4)]),
      api.coord([start, api.value(5)]),
    ];
    return { type: "polygon", shape: { points: points }, style: api.style() };
  }

  function groupBy(items, keyFn) {
    const grouped = {};
    items.forEach(function (item) {
      const key = keyFn(item);
      if (!grouped[key]) grouped[key] = [];
      grouped[key].push(item);
    });
    return grouped;
  }

  function zoomLinear(start, end, origin, factor, full, minSpan, dispatch) {
    const fullSpan = full.max - full.min;
    if (!(fullSpan > 0)) return;
    let span = (end - start) / factor;
    if (span >= fullSpan) {
      dispatch(full.min, full.max);
      return;
    }
    span = Math.max(span, minSpan || 0);
    const ratio = (end - start) === 0 ? 0.5 : (origin - start) / (end - start);
    let nextStart = origin - ratio * span;
    let nextEnd = nextStart + span;
    if (nextStart < full.min) {
      nextEnd += full.min - nextStart;
      nextStart = full.min;
    }
    if (nextEnd > full.max) {
      nextStart -= nextEnd - full.max;
      nextEnd = full.max;
    }
    nextStart = Math.max(full.min, nextStart);
    nextEnd = Math.min(full.max, nextEnd);
    dispatch(nextStart, nextEnd);
  }

  function shiftLinear(start, end, delta, full, dispatch) {
    const span = end - start;
    let nextStart = start + delta;
    let nextEnd = end + delta;
    const slack = span * 0.2;
    if (nextEnd < full.min + slack) {
      nextEnd = full.min + slack;
      nextStart = nextEnd - span;
    }
    if (nextStart > full.max - slack) {
      nextStart = full.max - slack;
      nextEnd = nextStart + span;
    }
    dispatch(nextStart, nextEnd);
  }

  function shiftLog(start, end, delta, fullMin, fullMax, dispatch) {
    const span = end - start;
    let nextStart = start + delta;
    let nextEnd = end + delta;
    const slack = span * 0.2;
    if (nextEnd < fullMin + slack) {
      nextEnd = fullMin + slack;
      nextStart = nextEnd - span;
    }
    if (nextStart > fullMax - slack) {
      nextStart = fullMax - slack;
      nextEnd = nextStart + span;
    }
    dispatch(nextStart, nextEnd);
  }

  function paddedPriceBounds(viewModel, log) {
    const prices = [];
    (viewModel.closes || []).forEach(function (bar) { prices.push(bar.close); });
    (viewModel.pivots || []).forEach(function (pivot) { prices.push(pivot.price); });
    (viewModel.boundaries || []).forEach(function (line) {
      (line.solid || []).concat(line.projection || []).forEach(function (pair) { prices.push(pair[1]); });
      if (line.breakout) prices.push(line.breakout.price);
    });
    (viewModel.zones || []).forEach(function (zone) {
      (zone.lower || []).concat(zone.upper || []).forEach(function (pair) { prices.push(pair[1]); });
      if (zone.breakout) prices.push(zone.breakout.price);
    });
    const finite = prices.filter(function (value) { return isFinite(value) && value > 0; });
    if (!finite.length) return { min: 1, max: 2 };
    const low = Math.min.apply(null, finite);
    const high = Math.max.apply(null, finite);
    if (log) {
      const span = Math.max(Math.log(high) - Math.log(low), 0.012);
      const pad = span * 0.18;
      return { min: Math.exp(Math.log(low) - pad), max: Math.exp(Math.log(high) + pad) };
    }
    const span = Math.max(high - low, Math.abs(high) * 0.01, 0.01);
    const pad = span * 0.18;
    return { min: low - pad, max: high + pad };
  }

  function formatPrice(value) {
    if (value == null || !isFinite(Number(value))) return "—";
    const number = Number(value);
    const abs = Math.abs(number);
    if (abs >= 10) return number.toFixed(2);
    if (abs >= 1) return number.toFixed(3);
    return number.toFixed(4);
  }

  function formatNum(value) {
    if (value == null || !isFinite(Number(value))) return "—";
    return Number(value).toFixed(3);
  }

  function esc(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  return { mount: mount };
})();
