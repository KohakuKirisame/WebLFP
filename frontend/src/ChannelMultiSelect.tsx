import { useEffect, useMemo, useRef, useState } from "react";

export default function ChannelMultiSelect({
  options,
  value,
  maxSelected,
  chinese,
  onChange,
}: {
  options: string[];
  value: string[];
  maxSelected: number;
  chinese: boolean;
  onChange: (channels: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const root = useRef<HTMLDivElement>(null);
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const filtered = useMemo(
    () => options.filter((channelId) => channelId.toLocaleLowerCase().includes(normalizedQuery)),
    [normalizedQuery, options],
  );

  function closeMenu() {
    setOpen(false);
    setQuery("");
  }

  useEffect(() => {
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) closeMenu();
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    return () => document.removeEventListener("mousedown", closeOnOutsideClick);
  }, []);

  function toggle(channelId: string) {
    if (value.includes(channelId)) {
      onChange(value.filter((item) => item !== channelId));
    } else if (value.length < maxSelected) {
      onChange([...value, channelId]);
    }
  }

  function remove(channelId: string) {
    onChange(value.filter((item) => item !== channelId));
  }

  return (
    <div className="channel-multiselect" ref={root}>
      <div className={`channel-control ${open ? "open" : ""}`} onClick={() => options.length && setOpen(true)}>
        <div className="channel-chips">
          {value.map((channelId) => (
            <span className="channel-chip" key={channelId}>
              <code>{channelId}</code>
              <button type="button" aria-label={`${chinese ? "移除通道" : "Remove channel"} ${channelId}`} onClick={(event) => {
                event.stopPropagation();
                remove(channelId);
              }}>×</button>
            </span>
          ))}
          <input
            aria-expanded={open}
            aria-label={chinese ? "搜索通道 ID" : "Search channel IDs"}
            disabled={options.length === 0}
            onFocus={() => options.length && setOpen(true)}
            onKeyDown={(event) => {
              if (event.key === "Escape") closeMenu();
              if (event.key === "Backspace" && !query && value.length) remove(value[value.length - 1]);
            }}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={options.length ? (chinese ? "搜索通道 ID…" : "Search channel IDs…") : (chinese ? "请先检查记录" : "Inspect the recording first")}
            value={query}
          />
        </div>
        <span className="channel-count">{value.length}/{maxSelected}</span>
      </div>
      {open && (
        <div className="channel-menu" role="listbox" aria-multiselectable="true">
          <div className="channel-menu-summary">
            <span>{chinese ? `${options.length} 个可用通道` : `${options.length} available channels`}</span>
            {value.length >= maxSelected && <strong>{chinese ? "已达到模型上限" : "Model limit reached"}</strong>}
          </div>
          <div className="channel-options">
            {filtered.length ? filtered.map((channelId) => {
              const selected = value.includes(channelId);
              return (
                <button
                  type="button"
                  className={selected ? "selected" : ""}
                  disabled={!selected && value.length >= maxSelected}
                  key={channelId}
                  role="option"
                  aria-selected={selected}
                  onClick={() => toggle(channelId)}
                >
                  <span className="channel-check">{selected ? "✓" : ""}</span>
                  <code>{channelId}</code>
                </button>
              );
            }) : <p>{chinese ? "没有匹配的通道 ID。" : "No channel IDs match this search."}</p>}
          </div>
        </div>
      )}
    </div>
  );
}
