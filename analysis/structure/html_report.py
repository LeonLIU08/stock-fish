"""把快照写成单文件 report.html。

页面、统计表和 view model 都读这份快照。这里不检测拐点，也不重新拟合。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from analysis.structure.config import PARAM_VERSION
from analysis.structure.methods.base import lookup_path
from analysis.structure.series import result_directory
from analysis.structure.view_model import (
    CONSTRAINT_LABEL,
    DIRECTION_LABEL,
    EVENT_LABEL,
    READING_NOTE,
    ROLE_LABEL,
    SCALE_DEFINITION_NOTE,
    SCALE_LABEL,
    STATUS_LABEL,
    TOKEN_LABEL,
    ZONE_LABEL,
    build_view_model,
    display_snapshot,
    zone_drawn,
)

_ASSET_DIR = Path(__file__).resolve().parent / "assets"

_PARAM_LABELS = {
    "version": "参数版本",
    "price_axis": "价格轴",
    "ewma_span": "波动 EWMA 跨度（根）",
    "vol_clip_multiple": "波动收益截断倍数",
    "volatility_floor": "波动下限",
    "touch_distance_ratio": "接触距离（波动倍数）",
    "touch_cluster_window": "接触簇窗口（根）",
    "min_touch_clusters": "趋势线最少接触簇",
    "zone_min_touch_clusters": "区间每侧最少接触簇",
    "zone_credit_full_span": "区间承认边界全长接触簇",
    "hard_break_ratio": "硬破坏深度（波动倍数）",
    "breakout_buffer_ratio": "突破缓冲（波动倍数）",
    "breakout_bars": "突破连续根数",
    "max_pivots_per_line": "单条线最多拐点",
    "channel_width_ratio_min": "通道宽度比下限",
    "channel_width_ratio_max": "通道宽度比上限",
    "convergence_width_ratio": "收敛宽度比",
    "sideways_normalized_slope": "横向斜率绝对值上限",
    "parallel_slope_gap": "平行斜率差上限",
    "expire_bars": "过期根数",
    "daily_gap_days": "日线间断（交易日）",
    "intraday_gap_median_multiple": "分钟线间断倍数",
    "max_zones_per_scale": "每尺度区间名额（主图不按它隐藏）",
}


def recognition_sections(snapshot: dict) -> dict:
    """某一天的识别表。调用方传入的快照已经是那天的拟合结果，这里不再按 as-of 过滤。"""
    counts = _recognition_counts(snapshot)
    return {
        "counts": counts,
        "pivots": _pivot_section(snapshot),
        "boundaries": _boundary_section(snapshot),
        "zones": _zone_section(snapshot),
        "methods": _method_sections(snapshot, snapshot),
        "events": _event_section(snapshot),
        "relations": _relation_section(snapshot),
        "appendix": _appendix(snapshot, snapshot),
        "phase": _phase_key(snapshot),
    }


def render_report(snapshot: dict, replay: dict | None = None) -> str:
    """生成完整 HTML。快照和 view model 都内嵌在文件里。

    传入 replay 时，图和识别表用打开时那一帧的前缀拟合。整段快照仍嵌在页面里，只作区间终点的留档。
    """
    scrub = None
    if replay:
        from analysis.structure.replay import compose_view_model, select_frame_index

        index = select_frame_index(replay["frames"], (snapshot.get("identity") or {}).get("as_of"))
        frame = replay["frames"][index]
        model = compose_view_model(replay, frame)
        sections = frame["sections"]
        scrub = frame
    else:
        shown = display_snapshot(snapshot)
        model = build_view_model(snapshot)
        sections = recognition_sections(shown)
    title = _title(snapshot)
    replay_script = ""
    if replay:
        from analysis.structure.replay import slim_replay

        replay_script = (
            f'<script type="application/json" id="structure-replay">{_json_for_script(slim_replay(replay))}</script>'
        )
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(title)}</title>",
        "<style>",
        _CSS,
        "</style>",
        "</head>",
        "<body>",
        _header(snapshot, sections, scrub),
        _reading(snapshot),
        _chart_section(snapshot, model, sections, scrub is not None),
        _quality_section(snapshot),
        _volatility_section(snapshot),
        sections["pivots"],
        sections["boundaries"],
        sections["zones"],
        f'<div id="method-sections">{sections["methods"]}</div>',
        sections["events"],
        sections["relations"],
        sections["appendix"],
        _params_section(snapshot),
        _glossary(),
        f'<script type="application/json" id="structure-snapshot">{_json_for_script(snapshot)}</script>',
        f'<script type="application/json" id="structure-view-model">{_json_for_script(model)}</script>',
        replay_script,
        "<script>",
        _asset("echarts.min.js"),
        "</script>",
        "<script>",
        _asset("structure-chart.js"),
        "</script>",
        "<script>",
        _BOOT,
        "</script>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(parts)


def write_result(output_root, snapshot: dict, replay: dict | None = None) -> dict:
    """把 snapshot.json 和 report.html 写进运行目录。同一身份再次运行会覆盖同名文件。"""
    identity = snapshot["identity"]
    directory = result_directory(
        output_root,
        identity["symbol"],
        identity["interval"],
        identity["price_axis"],
        identity["data_hash"],
        identity.get("param_version") or PARAM_VERSION,
    )
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_path = directory / "snapshot.json"
    report_path = directory / "report.html"
    snapshot_path.write_text(dump_snapshot(snapshot), encoding="utf-8")
    report_path.write_text(render_report(snapshot, replay), encoding="utf-8")
    return {"directory": directory, "snapshot": snapshot_path, "report": report_path}


def dump_snapshot(snapshot: dict) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _header(snapshot: dict, sections: dict, scrub: dict | None = None) -> str:
    identity = snapshot.get("identity") or {}
    version = identity.get("engine_version") or ""
    version_text = version or (identity.get("engine_version_note") or "引擎代码版本留空")
    chips = [
        _chip("标的", identity.get("symbol")),
        _chip("周期", identity.get("interval")),
        _chip("价格轴", identity.get("price_axis_label") or identity.get("price_axis")),
        _chip("参数", identity.get("param_version")),
        _chip("K 线", identity.get("bar_count")),
        _chip("引擎", version_text),
    ]
    if identity.get("data_source"):
        chips.append(_chip("数据来源", identity.get("data_source")))
    requested = _join_dates(identity.get("requested_start"), identity.get("requested_end"))
    actual = _join_dates(identity.get("start_time"), identity.get("end_time"), times=True)
    as_of = identity.get("as_of")
    history_note = identity.get("history_note") or ""
    banner = ""
    if history_note:
        banner = f'<p class="banner">{esc(history_note)}</p>'
    if scrub is not None:
        label = scrub.get("label") or "—"
        error = scrub.get("error")
        error_hidden = "" if error else " hidden"
        error_text = f"这一天没有拟合结果：{error}" if error else ""
        banner += (
            '<p class="banner" id="asof-banner">竖线在 '
            f'<span id="asof-date">{esc(label)}</span>。'
            "这一天的识别只用了截止到该根的 K 线。竖线右侧的价格只是对照，没有参与计算。</p>"
            f'<p class="banner" id="asof-error"{error_hidden}>{esc(error_text)}</p>'
        )
    elif as_of:
        banner += (
            '<p class="banner">页面按 as-of '
            f"{esc(fmt_time(as_of))} 显示：只保留当时已经可用的拐点、线和区间。"
            "收盘价仍是全部已加载 K 线。snapshot.json 保存的是整段运行，方便和这张图对照。</p>"
        )
    digest = identity.get("data_hash") or ""
    return f"""
<header class="topbar">
  <div class="topbar-row">
    <h1>{esc(_title(snapshot))}</h1>
    <p class="hash" title="{esc(digest)}">数据哈希 {esc(digest[:12] or "—")}</p>
  </div>
  <div class="chips">{"".join(chips)}</div>
  <nav>
    <a href="#chart-panel">图</a>
    <a href="#quality">数据质量</a>
    <a href="#volatility">波动与门槛</a>
    <a href="#pivots">拐点与线段</a>
    <a href="#boundaries">趋势线</a>
    <a href="#zones">趋势区间</a>
    {_method_nav(snapshot)}
    <a href="#events">事件</a>
    <a href="#relations">关系</a>
    <a href="#appendix">附录</a>
  </nav>
</header>
<section class="identity">
  <p><strong>请求区间</strong> {requested}</p>
  <p><strong>实际 K 线</strong> {actual}</p>
  {banner}
  <div class="counts">{_count_cards(snapshot, sections["counts"])}</div>
</section>
"""


def _reading(snapshot: dict) -> str:
    identity = snapshot.get("identity") or {}
    note = snapshot.get("reading") or READING_NOTE
    scale_note = snapshot.get("scale_definition_note") or SCALE_DEFINITION_NOTE
    definition = (identity.get("scale_definition") or {}).get("scales") or []
    rows = []
    for item in definition:
        name = SCALE_LABEL.get(item.get("name"), item.get("name"))
        rows.append(
            "<li><strong>"
            + esc(name)
            + "</strong> k="
            + esc(item.get("k"))
            + "，门槛下限 "
            + esc(item.get("delta_min"))
            + "，最大跨度 "
            + esc(item.get("max_span"))
            + " 根</li>"
        )
    scale_list = "<ul class=\"scale-list\">" + "".join(rows) + "</ul>" if rows else ""
    visible = "、".join(SCALE_LABEL.get(item, item) for item in identity.get("visible_scales") or ["mid"])
    return f"""
<section class="reading" id="reading">
  <p>{esc(note)}</p>
  <p>{esc(scale_note)}</p>
  {scale_list}
  <p class="muted">主图初次打开{esc(visible)}尺度。短、中、长都已经拟合，图例和上方按钮可以单独开关。</p>
</section>
"""


def _phase_key(shown: dict) -> str:
    """主图下方的阶段颜色。只列出这次图上出现过的主状态。"""
    record = ((shown or {}).get("methods") or {}).get("phase") or {}
    if record.get("status") not in (None, "completed"):
        return ""
    present = {
        item.get("category")
        for item in record.get("objects") or []
        if item.get("category")
    }
    chips = []
    for item in record.get("categories") or []:
        key = item.get("key")
        if key not in present:
            continue
        chips.append(
            "<li>"
            f'<span class="swatch" style="background:{esc(item.get("color") or "#8A8478")}"></span>'
            f"<span>{esc(item.get('name') or key)}</span>"
            "</li>"
        )
    if not chips:
        return ""
    return (
        '<p class="muted">阶段颜色。短、中、长用同一套。鼠标停在色块上也会显示这一段的名称。</p>'
        '<ul class="phase-key" aria-label="阶段颜色">'
        + "".join(chips)
        + "</ul>"
    )


def _zone_key(snapshot: dict) -> str:
    """图例旁边的三类区间定义。数字来自这次快照的参数。"""
    params = snapshot.get("params") or {}
    flat = _plain_num(params.get("sideways_normalized_slope"), 0.25)
    gap = _plain_num(params.get("parallel_slope_gap"), 0.35)
    low = _plain_num(params.get("channel_width_ratio_min"), 0.8)
    high = _plain_num(params.get("channel_width_ratio_max"), 1.25)
    narrow = _plain_num(params.get("convergence_width_ratio"), 0.8)
    items = [
        (
            "sideways",
            "横向区间",
            f"上下边界都接近水平，归一化斜率绝对值都小于 {flat}；"
            f"两条边界接近平行，归一化斜率差小于 {gap}；"
            f"宽度稳定，终点宽度是起点的 {low} 到 {high} 倍。",
        ),
        (
            "channel",
            "通道",
            f"两条边界接近平行，归一化斜率差小于 {gap}，"
            f"宽度稳定在起点的 {low} 到 {high} 倍。横向已经先判走。",
        ),
        (
            "convergence",
            "收敛区间",
            f"终点宽度不到起点的 {narrow} 倍，宽度仍然为正。"
            "横向和通道都没成立时，落到这一类。",
        ),
    ]
    rows = "".join(
        "<li>"
        f'<span class="swatch {kind}"></span>'
        f"<span><strong>{esc(title)}</strong> {esc(text)}</span>"
        "</li>"
        for kind, title, text in items
    )
    return (
        '<ul class="zone-key">'
        + rows
        + "</ul>"
        + '<p class="muted">宽度比是有效区间终点宽度除以起点宽度。'
        "归一化斜率用两条边界里较大的那个波动去除。"
        "同一尺度上，这三类主结果全部叠到主图，不再只留评分最高的三个。</p>"
    )


def _plain_num(value, default: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _chart_section(snapshot: dict, model: dict, sections: dict, scrub: bool = False) -> str:
    supports = int((sections.get("counts") or {}).get("supports") or 0)
    visible = set((snapshot.get("identity") or {}).get("visible_scales") or ["mid"])
    scale_buttons = []
    for scale in ("short", "mid", "long"):
        pressed = "true" if scale in visible else "false"
        scale_buttons.append(
            f'<button type="button" data-scale="{scale}" aria-pressed="{pressed}">{SCALE_LABEL[scale]}尺度</button>'
        )
    axis = model.get("axis_label") or ""
    if "zones" in (model.get("replaced_layers") or []):
        zone_sentence = (
            "区间形状打开时，第一版的三类区间色带默认关掉，图例里可以再打开对照。"
            "已经突破或过期的形状仍留在主图上，边线改成虚线。"
        )
    else:
        zone_sentence = (
            "下面三类区间都画在主图上，已经突破或过期的也保留，边线改成虚线。"
        )
    return f"""
<section class="panel" id="chart-panel">
  <div class="section-head">
    <h2>走势、趋势线与趋势区间</h2>
    <p id="validated-support-count" data-count="{supports}">已验证支撑 {supports} 条</p>
  </div>
  <div class="toolbar">
    <button type="button" id="reset-chart">复位</button>
    {"".join(scale_buttons)}
    <button type="button" data-extra="投影" aria-pressed="true">投影</button>
    <button type="button" data-extra="基础线段" aria-pressed="false">基础线段</button>
    <button type="button" data-extra="已过期" aria-pressed="false">已过期</button>
  </div>
  <p class="muted">纵轴是{esc(axis)}。趋势线在这根轴上是直线。</p>
  <div id="chart"></div>
  <div id="phase-key-slot">{sections.get("phase") or ""}</div>
  <p class="muted">横轴是 K 线序号，刻度是日期。滚轮以指针位置缩放，按住拖拽平移，底部滑块选择序号区间，双击图或按复位回到全部 K 线，已选中的趋势线或方向线段仍留在图上。点击表格里的一行，会在图上定位该对象。悬停显示实际股价；拐点同时显示极值时间和确认时间。鼠标停在主图下方的色块上，会显示这一段的名称。</p>
  {('<p class="muted">抓住 as-of 竖线左右拖动，吸附到某一根 K 线。拖动时只改图和日期，松开后更新下面的识别表。没有抓到竖线时，仍是缩放和平移。</p>' if scrub else '')}
  <p class="muted">图上默认是已验证的主结果。实线覆盖拟合用的序号，虚线是向后的投影。被收盘价突破的线保留到突破那一根，突破点用菱形标出。{zone_sentence}候选、拒绝、备选和其他边界对不进这张图。</p>
  {_zone_key(snapshot)}
</section>
"""


def _quality_section(snapshot: dict) -> str:
    quality = snapshot.get("quality") or {}
    gaps = quality.get("gaps") or []
    body = "".join(
        "<tr>"
        f"<td>{esc(item.get('index'))}</td>"
        f"<td>{esc(fmt_time(item.get('previous_time')))}</td>"
        f"<td>{esc(fmt_time(item.get('time')))}</td>"
        "</tr>"
        for item in gaps
    )
    table = ""
    if gaps:
        table = (
            '<div class="table-wrap"><table><thead><tr>'
            "<th>序号</th><th>前一根时间</th><th>后一根时间</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>"
        )
    return f"""
<section id="quality">
  <h2>数据质量</h2>
  <p>{esc(quality.get("grade_note") or "")}</p>
  <div class="table-wrap"><table>
    <thead><tr><th>K 线根数</th><th>间断次数</th><th>被拒绝的非正价格</th><th>质量等级</th></tr></thead>
    <tbody><tr>
      <td>{esc(quality.get("bar_count"))}</td>
      <td>{esc(quality.get("gap_count"))}</td>
      <td>{esc(quality.get("rejected_non_positive"))}</td>
      <td>{esc(quality.get("grade"))}</td>
    </tr></tbody>
  </table></div>
  {table}
  <p class="muted">非正收盘价会在进入引擎前整段拒绝，所以成功写出的快照里这一项是 0。日线里的周末和普通休市不算间断。</p>
</section>
"""


def _volatility_section(snapshot: dict) -> str:
    rows = ((snapshot.get("volatility") or {}).get("scales") or [])
    unit = (snapshot.get("volatility") or {}).get("unit") or ""
    body = []
    for item in rows:
        body.append(
            "<tr>"
            f"<td>{esc(SCALE_LABEL.get(item.get('scale'), item.get('scale')))}</td>"
            f"<td>{esc(item.get('k'))}</td>"
            f"<td>{esc(item.get('delta_min'))}</td>"
            f"<td>{esc(item.get('max_span'))}</td>"
            f"<td>{fmt_num(item.get('volatility_median'), 4)}</td>"
            f"<td>{fmt_num(item.get('threshold_median'), 4)}</td>"
            "</tr>"
        )
    note = rows[0].get("delta_min_note") if rows else ""
    return f"""
<section id="volatility">
  <h2>波动与门槛</h2>
  <p>样本内波动 s<sub>t</sub> 是截至当时的收益率波动，三个尺度共用这一条序列。尺度之间的差别在 k、门槛下限和最大跨度。实际使用的门槛在每一段开始时冻结。单位：{esc(unit) or "—"}。</p>
  <p class="muted">{esc(note or "")}</p>
  <div class="table-wrap"><table>
    <thead><tr>
      <th title="短、中、长，由参数包固定">尺度</th>
      <th title="反转门槛里的波动倍数">k</th>
      <th title="反转门槛下限。均匀轴会按段初收盘价换算">门槛下限</th>
      <th title="一条趋势线允许的最大序号跨度">最大跨度</th>
      <th title="全部已加载 K 线上 s_t 的中位数">s_t 中位数</th>
      <th title="该尺度已确认拐点上冻结门槛的中位数">实际门槛中位数</th>
    </tr></thead>
    <tbody>{"".join(body) or _empty_row(6)}</tbody>
  </table></div>
</section>
"""


def _pivot_section(shown: dict) -> str:
    stats = shown.get("pivot_stats") or []
    unit = ""
    body = []
    for item in stats:
        unit = item.get("change_unit") or unit
        body.append(
            "<tr>"
            f"<td>{esc(SCALE_LABEL.get(item.get('scale'), item.get('scale')))}</td>"
            f"<td>{esc(item.get('high_count'))}</td>"
            f"<td>{esc(item.get('low_count'))}</td>"
            f"<td>{fmt_num(item.get('confirm_delay_median'), 1)}</td>"
            f"<td>{esc(item.get('confirm_delay_max'))}</td>"
            f"<td>{esc(item.get('segment_count'))}</td>"
            f"<td>{fmt_num(item.get('span_median'), 1)}</td>"
            f"<td>{fmt_num(item.get('abs_change_median'), 4)}</td>"
            f"<td>{esc(item.get('temporary_count'))}</td>"
            "</tr>"
        )
    detail = []
    for item in list(shown.get("pivots") or []) + list(shown.get("temporary_pivots") or []):
        confirmed = item.get("confirm_index") is not None and item.get("available_time")
        kind = "pivot" if confirmed else "temporary"
        object_id = item.get("id") or f"{kind}:{item.get('scale')}:{item.get('role')}:{item.get('extreme_index')}"
        detail.append(
            f'<tr data-object-id="{esc(object_id)}">'
            f"<td>{esc(SCALE_LABEL.get(item.get('scale'), item.get('scale')))}</td>"
            f"<td>{esc(ROLE_LABEL.get(item.get('role'), item.get('role')))}</td>"
            f"<td>{esc(item.get('extreme_index'))}</td>"
            f"<td>{esc(fmt_time(item.get('extreme_time')))}</td>"
            f"<td>{esc(item.get('confirm_index'))}</td>"
            f"<td>{esc(fmt_time(item.get('confirm_time')))}</td>"
            f"<td>{esc(fmt_time(item.get('available_time')))}</td>"
            f"<td>{fmt_price(item.get('price'))}</td>"
            f"<td>{fmt_num(item.get('threshold'), 4)}</td>"
            "</tr>"
        )
    segments = []
    for item in shown.get("segments") or []:
        start = item.get("start") or {}
        end = item.get("end") or {}
        segments.append(
            f'<tr data-object-id="{esc(item.get("id"))}">'
            f"<td>{esc(item.get('id'))}</td>"
            f"<td>{esc(SCALE_LABEL.get(item.get('scale'), item.get('scale')))}</td>"
            f"<td>{esc(DIRECTION_LABEL.get(item.get('direction'), item.get('direction')))}</td>"
            f"<td>{esc(start.get('extreme_index'))} → {esc(end.get('extreme_index'))}</td>"
            f"<td>{esc(item.get('bar_span'))}</td>"
            f"<td>{fmt_num(item.get('change'), 4)}</td>"
            f"<td>{fmt_num(item.get('change_multiple'), 2)}</td>"
            f"<td>{esc(len(item.get('contains') or []))}</td>"
            "</tr>"
        )
    return f"""
<section id="pivots">
  <h2>拐点与线段</h2>
  <p>高点和低点只数已经确认的拐点。确认延迟是确认序号减去极值序号。基础线段连接相邻且角色交替的已确认拐点。涨跌幅在对数轴上是对数收益，在均匀轴上是价格差。当前单位：{esc(unit) or "—"}。</p>
  <div class="table-wrap"><table>
    <thead><tr>
      <th>尺度</th>
      <th title="已确认高点">高点</th>
      <th title="已确认低点">低点</th>
      <th title="确认序号减极值序号的中位数">确认延迟中位数</th>
      <th title="确认延迟的最大值">确认延迟最大</th>
      <th title="已确认基础线段条数">基础线段</th>
      <th title="线段跨越根数的中位数">跨越根数中位数</th>
      <th title="坐标涨跌幅绝对值的中位数">涨跌幅绝对值中位数</th>
      <th title="末端尚未确认的拐点，不进入高点和低点">临时拐点</th>
    </tr></thead>
    <tbody>{"".join(body) or _empty_row(9)}</tbody>
  </table></div>
  <details>
    <summary>拐点明细</summary>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>尺度</th><th>角色</th><th>极值序号</th><th>极值时间</th><th>确认序号</th><th>确认时间</th><th>可用时间</th><th>价格</th><th>门槛</th>
      </tr></thead>
      <tbody>{"".join(detail) or _empty_row(9)}</tbody>
    </table></div>
  </details>
  <details>
    <summary>基础线段明细</summary>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>标识</th><th>尺度</th><th>方向</th><th>极值序号</th><th>跨越根数</th><th>坐标涨跌幅</th><th>相对波动倍数</th><th>包含的更短线段</th>
      </tr></thead>
      <tbody>{"".join(segments) or _empty_row(8)}</tbody>
    </table></div>
  </details>
</section>
"""


def _boundary_section(shown: dict) -> str:
    primary = [item for item in shown.get("boundaries") or [] if item.get("primary")]
    return f"""
<section id="boundaries">
  <div class="section-head">
    <h2>趋势线</h2>
    <div class="filters" data-filter-for="boundary-table">
      <button type="button" data-mode="live" aria-pressed="true">已验证及之后</button>
      <button type="button" data-mode="all" aria-pressed="false">全部主结果</button>
      <button type="button" data-mode="validated" aria-pressed="false">仅已验证</button>
      <button type="button" data-mode="broken" aria-pressed="false">已突破</button>
    </div>
  </div>
  <p>每一行是一条主结果，按起点日期从早到晚排列。硬约束有一条没过，就不能标成已验证。软评分只出现在曾经通过硬约束的线上，分项和总分一起给出。简化惩罚在这一版是 0，上下文吻合标为未使用。备选线在附录里，不进主图。</p>
  {_line_table(primary, "boundary-table", "primary")}
  <p class="muted">点击一行，图会定位到这条线，并在线的前后各多显示两倍于这条线的走势。未成立的主结果默认先收起，可以用「全部主结果」打开。</p>
</section>
"""


def _zone_section(shown: dict) -> str:
    primary = [item for item in shown.get("zones") or [] if item.get("primary")]
    return f"""
<section id="zones">
  <div class="section-head">
    <h2>趋势区间</h2>
    <div class="filters" data-filter-for="zone-table">
      <button type="button" data-mode="live" aria-pressed="true">已验证及之后</button>
      <button type="button" data-mode="all" aria-pressed="false">全部主结果</button>
    </div>
  </div>
  <p>一个区间是同一尺度上的下边界和上边界，表按有效区间的起点日期从早到晚排列。有效区间是宽度为正、两侧都有接触、收盘价没有超过硬破坏深度的最长一段。横向、通道、收敛按图上的顺序判断。这三类主结果都画在主图上，突破和过期也保留。其他边界对留在表里。不同尺度的重叠不合并成一条证据。</p>
  {_zone_table(primary, "zone-table", "primary")}
</section>
"""


def _event_section(shown: dict) -> str:
    rows = []
    for item in shown.get("events") or []:
        cross = item.get("price_cross_index")
        rows.append(
            f'<tr data-object-id="{esc(item.get("object_id"))}">'
            f"<td>{esc(EVENT_LABEL.get(item.get('event_type'), item.get('event_type')))}</td>"
            f"<td>{esc(fmt_time(item.get('available_time')))}</td>"
            f"<td>{esc(item.get('event_index'))}</td>"
            f"<td>{fmt_price(item.get('reference_close'))}</td>"
            f"<td>{esc(item.get('object_id'))}</td>"
            f"<td>{esc(_object_kind_label(item.get('object_kind')))}</td>"
            f"<td>{esc(item.get('revision'))}</td>"
            f"<td class=\"wrap\">{esc(item.get('reason'))}</td>"
            f"<td>{esc(cross if cross is not None else '—')}</td>"
            "</tr>"
        )
    return f"""
<section id="events">
  <h2>事件</h2>
  <p>事件按系统可用时间排列。价格可以更早越过边界，那一列单独写「价格越过序号」，事件本身不早于结构可用的那一根。这些事件不是买卖指令。</p>
  <div class="table-wrap"><table id="event-table">
    <thead><tr>
      <th>事件类型</th>
      <th title="引擎当时可以使用这个结果的时间">可用时间</th>
      <th title="事件所在的 K 线序号">事件序号</th>
      <th>参考收盘价</th>
      <th>对象标识</th>
      <th>对象</th>
      <th>修订号</th>
      <th>原因</th>
      <th title="价格更早越过边界时的序号。没有则空">价格越过序号</th>
    </tr></thead>
    <tbody>{"".join(rows) or _empty_row(9)}</tbody>
  </table></div>
</section>
"""


def _relation_section(shown: dict) -> str:
    relations = shown.get("relations") or {}
    groups = []
    for item in relations.get("boundary_groups") or []:
        groups.append(_group_row(item, ROLE_LABEL.get(item.get("role"), item.get("role"))))
    for item in relations.get("zone_groups") or []:
        groups.append(_group_row(item, ZONE_LABEL.get(item.get("label"), item.get("label"))))
    overlaps = []
    for item in relations.get("overlaps") or []:
        ids = "、".join(item.get("zone_ids") or [])
        scales = "、".join(SCALE_LABEL.get(scale, scale) for scale in item.get("scales") or [])
        overlaps.append(
            "<tr>"
            f"<td class=\"wrap\">{esc(ids)}</td>"
            f"<td>{esc(scales)}</td>"
            f"<td>{esc(item.get('note') or relations.get('overlap_note'))}</td>"
            "</tr>"
        )
    contains = []
    for item in relations.get("segment_contains") or []:
        contains.append(
            "<tr>"
            f"<td>{esc(item.get('id'))}</td>"
            f"<td>{esc(SCALE_LABEL.get(item.get('scale'), item.get('scale')))}</td>"
            f"<td class=\"wrap\">{esc('、'.join(item.get('contains') or []))}</td>"
            "</tr>"
        )
    return f"""
<section id="relations">
  <h2>关系</h2>
  <p>去重之后，硬约束通过且软评分最高的一条是主结果，相近的其余线是备选。备选不产生之后的突破事件。{esc(relations.get("overlap_note") or "不合并为一条证据")}。</p>
  <h3>主结果与备选</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>主结果</th><th>尺度</th><th>角色或标签</th><th>备选</th></tr></thead>
    <tbody>{"".join(groups) or _empty_row(4)}</tbody>
  </table></div>
  <h3>跨尺度序号重叠</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>区间</th><th>尺度</th><th>说明</th></tr></thead>
    <tbody>{"".join(overlaps) or _empty_row(3)}</tbody>
  </table></div>
  <details>
    <summary>线段包含关系</summary>
    <div class="table-wrap"><table>
      <thead><tr><th>线段</th><th>尺度</th><th>包含的更短线段</th></tr></thead>
      <tbody>{"".join(contains) or _empty_row(3)}</tbody>
    </table></div>
  </details>
</section>
"""


def _method_nav(snapshot: dict) -> str:
    links = []
    for key, record in (snapshot.get("methods") or {}).items():
        links.append(f'<a href="#method-{esc(key)}">{esc((record or {}).get("name") or key)}</a>')
    return "".join(links)


def _method_sections(shown: dict, snapshot: dict) -> str:
    methods = shown.get("methods") or {}
    if not methods:
        return ""
    from analysis.structure.methods import registered_methods

    order = [method.key for method in registered_methods()]
    keys = [key for key in order if key in methods] + [key for key in methods if key not in order]
    return "".join(_method_section(key, methods[key], snapshot) for key in keys)


def _method_section(key: str, record: dict, snapshot: dict) -> str:
    from analysis.structure.methods import registered_methods

    method = next((item for item in registered_methods() if item.key == key), None)
    status = STATUS_LABEL.get(record.get("status"), record.get("status") or "")
    notes = "".join(f"<p>{esc(note)}</p>" for note in record.get("notes") or [])
    if key == "direction":
        notes = (
            "<p>同一尺度里，用已经确认的高点和低点给每根 K 线一个方向。"
            "未定不是类别。停滞只改标签，不改结构方向。</p>"
            + notes
        )
    reason = ""
    if record.get("status") != "completed" and record.get("reason"):
        reason = f'<p class="banner">{esc(record.get("reason"))}</p>'
    blocks = []
    for table in method.tables() if method else ():
        if table.appendix:
            continue
        blocks.append(_declared_table(key, table, _table_source(record, table), snapshot))
    return f"""
<section id="method-{esc(key)}">
  <h2>{esc(record.get("name") or key)}</h2>
  <p class="muted">{esc(record.get("section") or "")} · {esc(record.get("version") or "")} · {esc(status)}</p>
  {notes}
  {reason}
  {_method_params(record.get("params") or {})}
  {"".join(blocks)}
</section>
"""


def _method_params(params: dict) -> str:
    if not params:
        return ""
    labels = {
        "epsilon": "持平容差",
        "cross_buffer": "穿越缓冲",
        "cross_bars": "穿越连续根数",
        "stall_bars": "停滞根数（短 / 中 / 长）",
        "level_tolerance": "水平容差",
        "converging_width_ratio": "收敛宽度比",
        "diverging_width_ratio": "张开宽度比",
    }
    from analysis.structure.methods.pause import PARAM_LABELS
    from analysis.structure.methods.phase import PARAM_LABELS as PHASE_LABELS

    labels.update(PARAM_LABELS)
    labels.update(PHASE_LABELS)
    rows = []
    for key, value in params.items():
        if isinstance(value, dict):
            shown = "、".join(f"{TOKEN_LABEL.get(name, name)} {item}" for name, item in value.items())
        elif isinstance(value, bool):
            shown = "开" if value else "关"
        else:
            shown = value
        rows.append(f"<tr><td>{esc(labels.get(key, key))}</td><td>{esc(shown)}</td></tr>")
    return (
        '<div class="table-wrap"><table><thead><tr><th>参数</th><th>取值</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _method_appendix(shown: dict, snapshot: dict) -> str:
    methods = shown.get("methods") or {}
    if not methods:
        return ""
    from analysis.structure.methods import registered_methods

    blocks = []
    for method in registered_methods():
        record = methods.get(method.key)
        if not record:
            continue
        for table in method.tables():
            if not table.appendix:
                continue
            rows = _table_source(record, table)
            blocks.append(
                "<details><summary>"
                + esc(table.title)
                + f"（{len(rows)}）</summary>"
                + _declared_table(method.key, table, rows, snapshot)
                + "</details>"
            )
    return "".join(blocks)


def _table_source(record: dict, table) -> list:
    source = table.source or "objects"
    if source == "objects":
        return list(record.get("objects") or [])
    if source.startswith("extra."):
        return list((record.get("extra") or {}).get(source.split(".", 1)[1]) or [])
    return []


def _declared_table(method_key: str, table, rows: list, snapshot: dict) -> str:
    price_axis = ((snapshot.get("identity") or {}).get("price_axis")) or "log"
    head = "".join(f"<th>{esc(column.name)}</th>" for column in table.columns)
    body = []
    for item in rows:
        cells = "".join(
            f"<td>{_format_cell(lookup_path(item, column.path), column.format, price_axis)}</td>"
            for column in table.columns
        )
        object_id = item.get("id") or ""
        attr = f' data-object-id="{esc(object_id)}"' if object_id else ""
        body.append(f"<tr{attr}>{cells}</tr>")
    return (
        f'<div class="table-wrap"><table id="method-{esc(method_key)}-{esc(table.key)}">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body) or _empty_row(len(table.columns))}</tbody></table></div>"
    )


def _format_cell(value, fmt: str, price_axis: str) -> str:
    if value is None or value == "":
        return "—"
    if fmt == "time":
        return esc(fmt_time(value))
    if fmt == "price":
        return fmt_price(value)
    if fmt == "geometric":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return esc(value)
        if price_axis == "log":
            number = math.exp(number)
        return fmt_price(number)
    if fmt in ("multiple", "index"):
        return fmt_num(value, 4 if fmt == "multiple" else 0)
    if fmt == "status":
        return _status_badge(str(value))
    if fmt == "token":
        return esc(TOKEN_LABEL.get(value, value))
    return esc(value)


def _object_kind_label(kind) -> str:
    if kind == "boundary":
        return "趋势线"
    if kind == "zone":
        return "趋势区间"
    from analysis.structure.methods import registered_methods

    for method in registered_methods():
        if method.key == kind:
            return method.name
    return str(kind or "—")


def _appendix(shown: dict, snapshot: dict) -> str:
    alternates = [item for item in shown.get("boundaries") or [] if not item.get("primary")]
    zone_alternates = [item for item in shown.get("zones") or [] if not item.get("primary")]
    temporary = shown.get("temporary_segments") or []
    temp_rows = []
    for item in temporary:
        end = item.get("end") or {}
        temp_rows.append(
            "<tr>"
            f"<td>{esc(SCALE_LABEL.get(item.get('scale'), item.get('scale')))}</td>"
            f"<td>{esc(DIRECTION_LABEL.get(item.get('direction'), item.get('direction')))}</td>"
            f"<td>{esc(fmt_time(end.get('extreme_time')))}</td>"
            f"<td>{fmt_price(end.get('price'))}</td>"
            "</tr>"
        )
    return f"""
<section id="appendix">
  <h2>附录</h2>
  <details>
    <summary>备选趋势线（{len(alternates)}）</summary>
    {_line_table(alternates, "alternate-lines", "appendix")}
  </details>
  <details>
    <summary>备选趋势区间（{len(zone_alternates)}）</summary>
    {_zone_table(zone_alternates, "alternate-zones", "appendix")}
  </details>
  <details>
    <summary>临时末端（{len(temporary)}）</summary>
    <p class="muted">临时线段还没有走出反向门槛，不参与已验证趋势线和趋势区间。</p>
    <div class="table-wrap"><table>
      <thead><tr><th>尺度</th><th>方向</th><th>末端时间</th><th>末端价格</th></tr></thead>
      <tbody>{"".join(temp_rows) or _empty_row(4)}</tbody>
    </table></div>
  </details>
  {_method_appendix(shown, snapshot)}
</section>
"""


def _params_section(snapshot: dict) -> str:
    params = snapshot.get("params") or {}
    if not params:
        return ""
    rows = []
    for key, label in _PARAM_LABELS.items():
        if key not in params:
            continue
        rows.append(f"<tr><td>{esc(label)}</td><td>{esc(params.get(key))}</td></tr>")
    scales = params.get("scales") or []
    scale_rows = []
    for item in scales:
        scale_rows.append(
            "<tr>"
            f"<td>{esc(SCALE_LABEL.get(item.get('name'), item.get('name')))}</td>"
            f"<td>{esc(item.get('k'))}</td>"
            f"<td>{esc(item.get('delta_min'))}</td>"
            f"<td>{esc(item.get('max_span'))}</td>"
            "</tr>"
        )
    return f"""
<section id="params">
  <h2>参数包</h2>
  <p>这一版的阈值都封在 {esc(params.get("version") or PARAM_VERSION)} 里。以后要调数字，换参数版本，不覆盖这次的目录。</p>
  <div class="table-wrap"><table>
    <thead><tr><th>尺度</th><th>k</th><th>门槛下限</th><th>最大跨度</th></tr></thead>
    <tbody>{"".join(scale_rows)}</tbody>
  </table></div>
  <div class="table-wrap"><table>
    <thead><tr><th>参数</th><th>取值</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</section>
"""


def _glossary() -> str:
    return """
<section id="glossary">
  <h2>怎么读这些词</h2>
  <p>完整说明在 <span class="path">document/trend-structure-manual.md</span>。下面是页面上反复出现的词。</p>
  <dl class="glossary">
    <dt>极值时间</dt><dd>最高或最低收盘价所在的那根 K 线。</dd>
    <dt>确认时间</dt><dd>反向变化达到门槛的那根 K 线。确认之后，极值位置和价格不再改。</dd>
    <dt>系统可用时间</dt><dd>引擎第一次可以使用这个确认结果的时间。历史回放里它等于确认时间。</dd>
    <dt>基础线段</dt><dd>两个相邻已确认拐点之间的连线。第一版不把多条线段合并成一条。</dd>
    <dt>接触簇</dt><dd>贴着边界、序号间隔不超过 3 根的拐点并成一簇。已验证至少要有 3 个独立接触簇。</dd>
    <dt>硬约束</dt><dd>不过就不能标成已验证。软评分不能把没过硬约束的对象补成已验证。</dd>
    <dt>软评分</dt><dd>几何吻合、接触质量、相对波动的显著性、路径完整性、时间跨度，以及记为 0 的简化惩罚。总分是这些分项的平均。</dd>
    <dt>归一化斜率</dt><dd>每个序号的几何价格变化，除以这条线起点上冻结的波动尺度。</dd>
    <dt>投影</dt><dd>拟合序号之后，用同一条冻结直线向后延伸的虚线。它不是新的预测模型。</dd>
    <dt>突破</dt><dd>已验证之后，收盘价在边界错误一侧连续停留，并且深度超过缓冲。</dd>
    <dt>过期</dt><dd>有效区间结束后走过规定根数，仍没有突破，也没有新的验证。过期的趋势线默认收在「已过期」里。过期的趋势区间仍按原来的颜色留在主图上，边线改成虚线。</dd>
  </dl>
</section>
"""


def _line_table(rows: list, table_id: str, table_name: str) -> str:
    body = []
    for item in _by_start_time(rows):
        status = item.get("status") or ""
        role = item.get("role") or ""
        hidden = table_name == "primary" and status in {"candidate", "rejected"}
        score = item.get("score") or {}
        clusters = item.get("touch_clusters") or []
        body.append(
            "<tr "
            f'data-table="{table_name}" data-status="{esc(status)}" data-role="{esc(role)}" '
            f'data-object-id="{esc(item.get("id"))}"'
            + (" hidden" if hidden else "")
            + ">"
            f"<td>{esc(item.get('id'))}</td>"
            f"<td>{esc(item.get('revision'))}</td>"
            f"<td>{esc(SCALE_LABEL.get(item.get('scale'), item.get('scale')))}</td>"
            f"<td>{esc(ROLE_LABEL.get(role, role))}</td>"
            f"<td>{_status_badge(status)}</td>"
            f"<td class=\"wrap\">{esc(fmt_time(item.get('start_time')))} → {esc(fmt_time(item.get('end_time')))}"
            f"<br><span class=\"muted\">序号 {esc(item.get('start_index'))}–{esc(item.get('end_index'))}</span></td>"
            f"<td>{fmt_num(item.get('normalized_slope'))}</td>"
            f"<td>{esc(len(clusters))}</td>"
            f"<td>{esc(item.get('span'))}</td>"
            f"<td>{fmt_num(item.get('touch_error_median'))}</td>"
            f"<td>{fmt_num(item.get('touch_error_p90'))}</td>"
            f"<td>{fmt_num(item.get('max_break_depth'))}</td>"
            f"<td>{esc(item.get('longest_violation_bars'))}</td>"
            f"<td class=\"wrap\">{_constraint_badges(item.get('constraints'))}</td>"
            f"<td>{fmt_num((score or {}).get('total'))}</td>"
            f"<td>{fmt_num(score.get('geometry') if score else None)}</td>"
            f"<td>{fmt_num(score.get('touch_quality') if score else None)}</td>"
            f"<td>{fmt_num(score.get('significance') if score else None)}</td>"
            f"<td>{fmt_num(score.get('path_integrity') if score else None)}</td>"
            f"<td>{fmt_num(score.get('span') if score else None)}</td>"
            f"<td>{fmt_num(score.get('simplicity_penalty') if score else item.get('simplicity_penalty'))}</td>"
            f"<td>{esc(score.get('context_fit') if score else item.get('context_fit') or '—')}</td>"
            "</tr>"
        )
    return f"""
<div class="table-wrap"><table id="{esc(table_id)}">
  <thead><tr>
    <th>标识</th><th>修订</th><th>尺度</th><th>角色</th><th>状态</th>
    <th>起止</th>
    <th title="每个序号的几何价格变化，除以段初波动">归一化斜率</th>
    <th title="独立接触簇个数">接触簇</th>
    <th>序号跨度</th>
    <th title="接触误差中位数，单位是该段波动">接触误差中位</th>
    <th title="接触误差第 90 百分位">接触误差 P90</th>
    <th title="最大破坏深度，单位是该段波动">最大破坏</th>
    <th>最长越界根数</th>
    <th>硬约束</th>
    <th>总分</th>
    <th title="点到直线的贴合">几何吻合</th>
    <th title="接触簇的质量和间距">接触质量</th>
    <th title="涨跌相对波动有多大">显著性</th>
    <th title="有效路径里没有大的破坏">路径完整性</th>
    <th title="跨度相对该尺度上限">时间跨度</th>
    <th title="这一版没有合并内部线段，固定为 0">简化惩罚</th>
    <th title="这一版没有使用上下文吻合">上下文</th>
  </tr></thead>
  <tbody>{"".join(body) or _empty_row(22)}</tbody>
</table></div>
"""


def _zone_table(rows: list, table_id: str, table_name: str) -> str:
    body = []
    for item in _by_start_time(rows):
        status = item.get("status") or ""
        hidden = table_name == "primary" and status in {"candidate", "rejected"}
        score = item.get("score") or {}
        chart = "主图" if zone_drawn(item) else "主图不画"
        body.append(
            "<tr "
            f'data-table="{table_name}" data-status="{esc(status)}" data-role="zone" '
            f'data-object-id="{esc(item.get("id"))}"'
            + (" hidden" if hidden else "")
            + ">"
            f"<td class=\"wrap\">{esc(item.get('lower_id'))}<br>{esc(item.get('upper_id'))}</td>"
            f"<td>{esc(SCALE_LABEL.get(item.get('scale'), item.get('scale')))}</td>"
            f"<td>{esc(ZONE_LABEL.get(item.get('label'), item.get('label')))}</td>"
            f"<td>{_status_badge(status)}</td>"
            f"<td>{esc(chart)}</td>"
            f"<td class=\"wrap\">{esc(fmt_time(item.get('start_time')))} → {esc(fmt_time(item.get('end_time')))}"
            f"<br><span class=\"muted\">序号 {esc(item.get('effective_start'))}–{esc(item.get('effective_end'))}</span></td>"
            f"<td>{fmt_num(item.get('start_width'), 4)}</td>"
            f"<td>{fmt_num(item.get('end_width'), 4)}</td>"
            f"<td>{fmt_num(item.get('width_ratio'))}</td>"
            f"<td>{fmt_num(item.get('slope_gap'), 4)}</td>"
            f"<td>{fmt_num(item.get('normalized_slope_gap'))}</td>"
            f"<td>{esc(item.get('lower_touch_clusters'))}</td>"
            f"<td>{esc(item.get('upper_touch_clusters'))}</td>"
            f"<td>{fmt_num(item.get('max_break_depth'))}</td>"
            f"<td class=\"wrap\">{_constraint_badges(item.get('constraints'))}</td>"
            f"<td>{fmt_num(score.get('total') if score else None)}</td>"
            f"<td>{fmt_num(score.get('geometry') if score else None)}</td>"
            f"<td>{fmt_num(score.get('touch_quality') if score else None)}</td>"
            f"<td>{fmt_num(score.get('significance') if score else None)}</td>"
            f"<td>{fmt_num(score.get('path_integrity') if score else None)}</td>"
            f"<td>{fmt_num(score.get('span') if score else None)}</td>"
            f"<td>{esc(score.get('context_fit') if score else item.get('context_fit') or '—')}</td>"
            "</tr>"
        )
    return f"""
<div class="table-wrap"><table id="{esc(table_id)}">
  <thead><tr>
    <th>上下边界</th><th>尺度</th><th>几何标签</th><th>状态</th><th>主图</th>
    <th>有效区间</th>
    <th title="有效区间起点的几何宽度">起始宽度</th>
    <th title="有效区间终点的几何宽度">结束宽度</th>
    <th title="结束宽度除以起始宽度">宽度变化比</th>
    <th title="上下斜率之差的绝对值">斜率差</th>
    <th title="斜率差除以两条线里较大的波动">归一化斜率差</th>
    <th>下侧接触簇</th><th>上侧接触簇</th>
    <th>区间内最大破坏</th>
    <th>硬约束</th><th>总分</th>
    <th>几何吻合</th><th>接触质量</th><th>显著性</th><th>路径完整性</th><th>时间跨度</th><th>上下文</th>
  </tr></thead>
  <tbody>{"".join(body) or _empty_row(22)}</tbody>
</table></div>
"""


def _recognition_counts(shown: dict) -> dict:
    boundaries = shown.get("boundaries") or []
    zones = shown.get("zones") or []
    events = shown.get("events") or []
    return {
        "pivots": len(shown.get("pivots") or []),
        "lines": sum(1 for item in boundaries if item.get("primary") and item.get("status") == "validated"),
        "zones": sum(1 for item in zones if item.get("primary") and item.get("status") == "validated"),
        "breakouts": sum(1 for item in events if item.get("event_type") == "breakout"),
        "supports": len(_validated_supports(shown)),
    }


def _count_cards(snapshot: dict, counts: dict) -> str:
    quality = snapshot.get("quality") or {}
    cards = [
        ("K 线", (snapshot.get("identity") or {}).get("bar_count"), None),
        ("间断", quality.get("gap_count"), None),
        ("已确认拐点", counts.get("pivots"), "count-pivots"),
        ("已验证趋势线", counts.get("lines"), "count-lines"),
        ("已验证区间", counts.get("zones"), "count-zones"),
        ("突破事件", counts.get("breakouts"), "count-breakouts"),
    ]
    pieces = []
    for label, value, node_id in cards:
        ident = f' id="{node_id}"' if node_id else ""
        pieces.append(f'<div class="card"><span>{esc(label)}</span><strong{ident}>{esc(value)}</strong></div>')
    return "".join(pieces)


def _group_row(item: dict, kind: str) -> str:
    alternates = "、".join(item.get("alternate_ids") or [])
    return (
        "<tr>"
        f"<td>{esc(item.get('primary_id'))}</td>"
        f"<td>{esc(SCALE_LABEL.get(item.get('scale'), item.get('scale')))}</td>"
        f"<td>{esc(kind)}</td>"
        f"<td class=\"wrap\">{esc(alternates)}</td>"
        "</tr>"
    )


def _constraint_badges(checks) -> str:
    if not checks:
        return "—"
    parts = []
    for check in checks:
        passed = bool(check.get("passed"))
        label = CONSTRAINT_LABEL.get(check.get("name"), check.get("name"))
        parts.append(
            f'<span class="badge {"ok" if passed else "bad"}" title="{esc(check.get("detail") or "")}">'
            f'{esc(label)} {"通过" if passed else "失败"}</span>'
        )
    return " ".join(parts)


def _status_badge(status: str) -> str:
    label = STATUS_LABEL.get(status, status)
    return f'<span class="badge status-{esc(status)}">{esc(label)}</span>'


def _chip(label: str, value) -> str:
    return f'<span class="chip"><em>{esc(label)}</em>{esc(value if value not in (None, "") else "—")}</span>'


def _validated_supports(shown: dict) -> list:
    return [
        item
        for item in shown.get("boundaries") or []
        if item.get("primary") and item.get("status") == "validated" and item.get("role") == "support"
    ]


def _title(snapshot: dict) -> str:
    identity = snapshot.get("identity") or {}
    symbol = identity.get("symbol") or "趋势结构"
    interval = identity.get("interval") or ""
    return f"{symbol} {interval} 趋势结构".strip()


def _join_dates(start, end, times: bool = False) -> str:
    if times:
        left = fmt_time(start)
        right = fmt_time(end)
    else:
        left = start or "—"
        right = end or "—"
    return f"{esc(left)} 至 {esc(right)}"


def _by_start_time(rows: list) -> list:
    """页面表格按起点时间从早到晚排。同一天再按起点序号和标识排，顺序稳定。"""
    return sorted(rows, key=_start_sort_key)


def _start_sort_key(item: dict) -> tuple:
    start = str(item.get("start_time") or "")
    index = item.get("start_index")
    if index is None:
        index = item.get("effective_start")
    try:
        index_value = int(index)
    except (TypeError, ValueError):
        index_value = 0
    return (start, index_value, str(item.get("id") or ""))


def _empty_row(columns: int) -> str:
    return f'<tr><td colspan="{columns}">这一节没有记录</td></tr>'


def _json_for_script(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _asset(name: str) -> str:
    path = _ASSET_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少图表资源 {path}")
    return path.read_text(encoding="utf-8")


def esc(value) -> str:
    text = "—" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_time(value) -> str:
    if not value:
        return "—"
    text = str(value).replace("T", " ")
    if "." in text:
        text = text.split(".", 1)[0]
    offset = text.rfind("+")
    if offset > 10:
        text = text[:offset]
    if text.endswith("Z"):
        text = text[:-1]
    if text.endswith(" 00:00:00"):
        return text[:10]
    return text[:16] if len(text) >= 16 else text


def fmt_num(value, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    return f"{number:.{digits}f}"


def fmt_price(value) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    return f"{number:.4f}"


_CSS = """
:root {
  color-scheme: dark;
  --bg: #14120f;
  --panel: #1d1a16;
  --panel-2: #262017;
  --text: #f3efe4;
  --muted: #a79f91;
  --line: #3a342c;
  --accent: #e0a45a;
  --ok: #8fce93;
  --bad: #f07167;
  --info: #7eb6ff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.55 "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Noto Sans SC", sans-serif;
}
h1, h2, h3 {
  font-family: "Iowan Old Style", Palatino, "Palatino Linotype", "Songti SC", serif;
  font-weight: 600;
  letter-spacing: 0;
}
h1 { font-size: 1.55rem; margin: 0; }
h2 { font-size: 1.3rem; margin: 0 0 0.4rem; }
h3 { font-size: 1.02rem; margin: 1.1rem 0 0.4rem; }
a { color: var(--accent); text-decoration: none; }
p { margin: 0.35rem 0 0.7rem; }
.topbar {
  position: sticky;
  top: 0;
  z-index: 3;
  padding: 0.85rem 1.25rem 0.7rem;
  background: rgba(20, 18, 15, 0.94);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(12px);
}
section[id], .identity { scroll-margin-top: 9.5rem; }
.topbar-row, .section-head, .toolbar, .chips, nav, .counts, .filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 0.8rem;
  align-items: center;
}
.topbar-row, .section-head { justify-content: space-between; }
.hash, .muted, .chip em { color: var(--muted); }
.zone-key { list-style: none; padding: 0; margin: 0.2rem 0 0.8rem; }
.zone-key li { display: flex; gap: 0.6rem; align-items: flex-start; margin: 0.35rem 0; }
.phase-key {
  list-style: none;
  padding: 0;
  margin: 0.15rem 0 0.7rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.9rem;
}
.phase-key li { display: inline-flex; gap: 0.4rem; align-items: center; color: #c8c2b4; }
.phase-key .swatch { margin-top: 0; }
.lane-tip {
  position: absolute;
  z-index: 4;
  pointer-events: none;
  padding: 0.18rem 0.45rem;
  border: 1px solid #4a4338;
  border-radius: 6px;
  background: #262017;
  color: #f3efe4;
  font-size: 12px;
  line-height: 1.4;
  white-space: nowrap;
}
.lane-tip[hidden] { display: none; }
.swatch {
  width: 0.9rem;
  height: 0.9rem;
  margin-top: 0.28rem;
  border-radius: 2px;
  flex: 0 0 auto;
}
.swatch.channel { background: rgba(126, 182, 255, 0.45); box-shadow: inset 0 0 0 1px rgba(126, 182, 255, 0.9); }
.swatch.convergence { background: rgba(196, 148, 255, 0.45); box-shadow: inset 0 0 0 1px rgba(196, 148, 255, 0.9); }
.swatch.sideways { background: rgba(240, 176, 96, 0.45); box-shadow: inset 0 0 0 1px rgba(240, 176, 96, 0.9); }
nav a { margin-right: 0.75rem; }
.chips { margin: 0.55rem 0; }
.chip, .card, button {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 999px;
}
.chip { padding: 0.15rem 0.7rem; }
.chip em { font-style: normal; margin-right: 0.4rem; }
.identity, section { padding: 0.4rem 1.25rem 1rem; }
.reading, .panel, .banner {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.9rem 1rem;
}
.reading { border-left: 4px solid var(--accent); }
.banner { margin-top: 0.8rem; background: #2a2418; }
.scale-list { margin: 0.2rem 0 0.6rem; padding-left: 1.2rem; }
.counts { margin-top: 0.8rem; }
.card {
  min-width: 7rem;
  border-radius: 14px;
  padding: 0.45rem 0.75rem;
  display: flex;
  flex-direction: column;
}
.card span { color: var(--muted); font-size: 0.82rem; }
.card strong { font-size: 1.25rem; }
.toolbar { margin-bottom: 0.7rem; }
button {
  color: var(--text);
  padding: 0.28rem 0.75rem;
  cursor: pointer;
  font: inherit;
}
button[aria-pressed="true"] {
  background: #3a3124;
  border-color: var(--accent);
  color: #f6e2c2;
}
#chart { position: relative; width: 100%; height: min(68vh, 640px); min-height: 420px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }
table { border-collapse: collapse; width: max-content; min-width: 100%; }
th, td {
  padding: 0.45rem 0.65rem;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
  vertical-align: top;
}
th { color: var(--muted); font-weight: 600; background: #221e19; position: sticky; top: 0; }
td.wrap { white-space: normal; min-width: 14rem; }
tr[data-object-id] { cursor: pointer; }
tr[data-object-id]:hover, tr.is-selected { background: #322a20; }
tr[data-status="validated"] { box-shadow: inset 3px 0 0 var(--ok); }
tr[data-status="broken"] { box-shadow: inset 3px 0 0 var(--bad); }
tr[data-status="expired"] { box-shadow: inset 3px 0 0 var(--muted); }
.badge {
  display: inline-block;
  margin: 0.1rem 0.2rem 0.1rem 0;
  padding: 0.05rem 0.4rem;
  border-radius: 999px;
  border: 1px solid var(--line);
  font-size: 0.78rem;
}
.badge.ok, .status-validated { color: var(--ok); }
.badge.bad, .status-rejected { color: var(--bad); }
.status-broken { color: #ff8f9a; }
.status-candidate, .status-expired { color: var(--muted); }
details { margin: 0.8rem 0; }
summary { cursor: pointer; color: #f6e2c2; }
.glossary { display: grid; grid-template-columns: 8rem 1fr; gap: 0.35rem 0.8rem; }
.glossary dt { color: var(--accent); }
.glossary dd { margin: 0; }
.path { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #f6e2c2; }
.filters { margin-bottom: 0.4rem; }
@media (max-width: 720px) {
  .glossary { grid-template-columns: 1fr; }
  #chart { min-height: 360px; }
  h1 { font-size: 1.25rem; }
}
"""

_BOOT = """
var viewModel = JSON.parse(document.getElementById("structure-view-model").textContent);
var replayNode = document.getElementById("structure-replay");
var replay = replayNode ? JSON.parse(replayNode.textContent) : null;
var chartRoot = document.getElementById("chart");
var api = TrendChart.mount(chartRoot, viewModel, replay);
var resetButton = document.getElementById("reset-chart");
if (resetButton) resetButton.addEventListener("click", function () { api.reset(); });
var scaleButtons = Array.prototype.slice.call(document.querySelectorAll("[data-scale]"));
function pressedScales() {
  return scaleButtons.filter(function (button) {
    return button.getAttribute("aria-pressed") === "true";
  }).map(function (button) { return button.getAttribute("data-scale"); });
}
function paint(state) {
  scaleButtons.forEach(function (button) {
    var on = state.scales.indexOf(button.getAttribute("data-scale")) >= 0;
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
  document.querySelectorAll("[data-extra]").forEach(function (button) {
    var name = button.getAttribute("data-extra");
    var on = name === "投影" ? state.projection : (name === "基础线段" ? state.segments : state.expired);
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
}
scaleButtons.forEach(function (button) {
  button.addEventListener("click", function () {
    var on = button.getAttribute("aria-pressed") === "true";
    button.setAttribute("aria-pressed", on ? "false" : "true");
    api.setScales(pressedScales());
  });
});
document.querySelectorAll("[data-extra]").forEach(function (button) {
  button.addEventListener("click", function () {
    var on = button.getAttribute("aria-pressed") !== "true";
    button.setAttribute("aria-pressed", on ? "true" : "false");
    api.setExtra(button.getAttribute("data-extra"), on);
  });
});
api.onChange(paint);
paint(api.state());
document.body.addEventListener("click", function (event) {
  var row = event.target.closest ? event.target.closest("tr[data-object-id]") : null;
  if (!row) return;
  document.querySelectorAll("tr.is-selected").forEach(function (item) { item.classList.remove("is-selected"); });
  row.classList.add("is-selected");
  api.focus(row.getAttribute("data-object-id"));
  var panel = document.getElementById("chart-panel");
  if (panel) panel.scrollIntoView({ behavior: "smooth", block: "center" });
});
function bindFilters(root) {
  (root || document).querySelectorAll("[data-filter-for]").forEach(function (group) {
    if (group.getAttribute("data-bound") === "true") return;
    group.setAttribute("data-bound", "true");
    var table = document.getElementById(group.getAttribute("data-filter-for"));
    group.querySelectorAll("[data-mode]").forEach(function (button) {
      button.addEventListener("click", function () {
        group.querySelectorAll("[data-mode]").forEach(function (item) {
          item.setAttribute("aria-pressed", item === button ? "true" : "false");
        });
        var mode = button.getAttribute("data-mode");
        table.querySelectorAll("tbody tr").forEach(function (row) {
          var status = row.getAttribute("data-status");
          if (!status) return;
          if (mode === "all") row.hidden = false;
          else if (mode === "live") row.hidden = ["validated", "broken", "expired"].indexOf(status) < 0;
          else row.hidden = status !== mode;
        });
      });
    });
  });
}
bindFilters(document);
function replaceSection(id, html) {
  if (html == null) return;
  var node = document.getElementById(id);
  if (!node) return;
  node.outerHTML = html;
  bindFilters(document.getElementById(id) || document);
}
function setCount(id, value) {
  var node = document.getElementById(id);
  if (node) node.textContent = value;
}
if (replay && api.onAsOf) {
  api.onAsOf(function (frame, committed) {
    var dateNode = document.getElementById("asof-date");
    if (dateNode) dateNode.textContent = frame.label || "—";
    if (!committed) return;
    var errorNode = document.getElementById("asof-error");
    if (errorNode) {
      errorNode.hidden = !frame.error;
      errorNode.textContent = frame.error ? ("这一天没有拟合结果：" + frame.error) : "";
    }
    var sections = frame.sections || {};
    ["pivots", "boundaries", "zones", "events", "relations", "appendix"].forEach(function (id) {
      replaceSection(id, sections[id]);
    });
    var methods = document.getElementById("method-sections");
    if (methods && sections.methods != null) methods.innerHTML = sections.methods;
    var phase = document.getElementById("phase-key-slot");
    if (phase && sections.phase != null) phase.innerHTML = sections.phase;
    var counts = frame.counts || {};
    setCount("count-pivots", counts.pivots);
    setCount("count-lines", counts.lines);
    setCount("count-zones", counts.zones);
    setCount("count-breakouts", counts.breakouts);
    var support = document.getElementById("validated-support-count");
    if (support) {
      support.textContent = "已验证支撑 " + (counts.supports || 0) + " 条";
      support.setAttribute("data-count", String(counts.supports || 0));
    }
  });
}
"""
