(function () {
  const PITCH_WIDTH = 105;
  const PITCH_HEIGHT = 68;
  const PLAYER_RADIUS = 1.4;
  const ARROW_LENGTH = 0.75;
  const THETA_STEP = 0.15;
  const API_BASE =
    window.location.hostname === "127.0.0.1"
      ? "http://127.0.0.1:8000"
      : "http://localhost:8000";
  const EPV_DEBOUNCE_MS = 200;

  const pitchEl = document.getElementById("pitch");
  const playersLayer = document.getElementById("players-layer");
  const thetaHint = document.getElementById("theta-hint");
  const thetaControls = document.getElementById("theta-controls");
  const thetaSlider = document.getElementById("theta-slider");
  const thetaValue = document.getElementById("theta-value");
  const thetaLeftBtn = document.getElementById("theta-left");
  const thetaRightBtn = document.getElementById("theta-right");
  const epvStatusEl = document.getElementById("epv-status");
  const epvValueEl = document.getElementById("epv-value");
  const bestActionEl = document.getElementById("best-action");
  const qShootEl = document.getElementById("q-shoot");
  const qPassEl = document.getElementById("q-pass");
  const qDribbleEl = document.getElementById("q-dribble");
  const bestActionReasonEl = document.getElementById("best-action-reason");
  const epvActionDetailEl = document.getElementById("epv-action-detail");
  const epvProfileInfoEl = document.getElementById("epv-profile-info");
  const passVizLayer = document.getElementById("pass-viz-layer");

  function rad2deg(rad) {
    return (rad * 180) / Math.PI;
  }

  function createInitialPlayers() {
    const players = [];
    let id = 1;
    for (let i = 0; i < 10; i++) {
      players.push({
        id: "home-" + id,
        team: "home",
        x: 15 + (i % 5) * 8,
        y: 8 + Math.floor(i / 5) * 25 + (i < 5 ? 0 : 20),
        theta: 0,
        hasBall: i === 2,
      });
      id++;
    }
    id = 1;
    for (let i = 0; i < 10; i++) {
      players.push({
        id: "away-" + id,
        team: "away",
        x: 90 - (i % 5) * 8,
        y: 8 + Math.floor(i / 5) * 25 + (i < 5 ? 0 : 20),
        theta: 0,
        hasBall: false,
      });
      id++;
    }
    if (!players.some((p) => p.hasBall)) players[2].hasBall = true;
    return players;
  }

  function get433Players() {
    const homeY = [10, 22, 34, 46, 18, 34, 50, 22, 34, 46];
    const homeX = [8, 8, 8, 8, 28, 28, 28, 48, 48, 48];
    const awayY = [10, 22, 34, 46, 18, 34, 50, 22, 34, 46];
    const awayX = [97, 97, 97, 97, 77, 77, 77, 57, 57, 57];
    const players = [];
    for (let i = 0; i < 10; i++) {
      players.push({
        id: "home-" + (i + 1),
        team: "home",
        x: homeX[i],
        y: homeY[i],
        theta: 0,
        hasBall: i === 5,
      });
    }
    for (let i = 0; i < 10; i++) {
      players.push({
        id: "away-" + (i + 1),
        team: "away",
        x: awayX[i],
        y: awayY[i],
        theta: 0,
        hasBall: false,
      });
    }
    return players;
  }

  function get442Players() {
    const homeY = [10, 22, 34, 46, 12, 26, 42, 56, 28, 40];
    const homeX = [8, 8, 8, 8, 28, 28, 28, 28, 48, 48];
    const awayY = [10, 22, 34, 46, 12, 26, 42, 56, 28, 40];
    const awayX = [97, 97, 97, 97, 77, 77, 77, 77, 57, 57];
    const players = [];
    for (let i = 0; i < 10; i++) {
      players.push({
        id: "home-" + (i + 1),
        team: "home",
        x: homeX[i],
        y: homeY[i],
        theta: 0,
        hasBall: i === 5,
      });
    }
    for (let i = 0; i < 10; i++) {
      players.push({
        id: "away-" + (i + 1),
        team: "away",
        x: awayX[i],
        y: awayY[i],
        theta: 0,
        hasBall: false,
      });
    }
    return players;
  }

  function resetToFormation(getPlayers) {
    state.players = getPlayers();
    state.playerProfileIds = {};
    state.profilePopoverPlayerId = null;
    hideProfilePopover();
    var profileEl = document.getElementById("ball-owner-profile");
    if (profileEl) profileEl.value = "average";
    renderPlayers();
    updateThetaUI();
    requestEpvUpdate();
  }

  let state = {
    players: createInitialPlayers(),
    playerProfiles: [],
    playerProfileIds: {},
    profilePopoverPlayerId: null,
    drag: null,
    epvDebounceTimer: null,
  };

  function loadPlayerProfiles() {
    fetch(API_BASE + "/players")
      .then(function (res) { return res.ok ? res.json() : []; })
      .then(function (list) {
        state.playerProfiles = Array.isArray(list) ? list : [];
        const sel = document.getElementById("ball-owner-profile");
        if (!sel) return;
        while (sel.options.length > 1) sel.remove(1);
        state.playerProfiles.forEach(function (pro) {
          const opt = document.createElement("option");
          opt.value = String(pro.player_id);
          opt.textContent = pro.display_name || "Player " + pro.player_id;
          sel.appendChild(opt);
        });
      })
      .catch(function () { state.playerProfiles = []; });
  }

  function getBallOwner() {
    return state.players.find((p) => p.hasBall) || null;
  }

  function getProfilePopoverEl() {
    return document.getElementById("profile-popover");
  }

  function getPerPlayerProfileSelect() {
    return document.getElementById("per-player-profile");
  }

  function hideProfilePopover() {
    const el = getProfilePopoverEl();
    if (el) {
      el.classList.add("hidden");
    }
    state.profilePopoverPlayerId = null;
  }

  function showProfilePopover(player) {
    const pop = getProfilePopoverEl();
    const sel = getPerPlayerProfileSelect();
    if (!pop || !sel) return;
    state.profilePopoverPlayerId = player.id;
    pop.style.left = (player.x / 105) * 100 + "%";
    pop.style.top = ((player.y + 5) / 68) * 100 + "%";
    sel.innerHTML = "";
    const optAverage = document.createElement("option");
    optAverage.value = "average";
    optAverage.textContent = "Average";
    sel.appendChild(optAverage);
    state.playerProfiles.forEach(function (pro) {
      const opt = document.createElement("option");
      opt.value = String(pro.player_id);
      opt.textContent = pro.display_name || pro.label || "Player " + pro.player_id;
      sel.appendChild(opt);
    });
    sel.value = state.playerProfileIds[player.id] || "average";
    pop.classList.remove("hidden");
    sel.focus();
  }

  function setupProfilePopoverListeners() {
    const sel = getPerPlayerProfileSelect();
    if (!sel) return;
    sel.addEventListener("change", function () {
      const pid = state.profilePopoverPlayerId;
      if (!pid) return;
      const val = sel.value;
      if (val === "average") {
        delete state.playerProfileIds[pid];
      } else {
        state.playerProfileIds[pid] = val;
      }
      requestEpvUpdate();
    });
  }

  function clientToPitch(clientX, clientY) {
    const rect = pitchEl.getBoundingClientRect();
    const scale = Math.min(rect.width / PITCH_WIDTH, rect.height / PITCH_HEIGHT);
    const offsetX = (rect.width - PITCH_WIDTH * scale) / 2;
    const offsetY = (rect.height - PITCH_HEIGHT * scale) / 2;
    const x = (clientX - rect.left - offsetX) / scale;
    const y = (clientY - rect.top - offsetY) / scale;
    return { x, y };
  }

  function clampToPitch(x, y) {
    return {
      x: Math.max(PLAYER_RADIUS, Math.min(PITCH_WIDTH - PLAYER_RADIUS, x)),
      y: Math.max(PLAYER_RADIUS, Math.min(PITCH_HEIGHT - PLAYER_RADIUS, y)),
    };
  }

  function normalizeTheta(theta) {
    let t = theta;
    while (t > Math.PI) t -= 2 * Math.PI;
    while (t < -Math.PI) t += 2 * Math.PI;
    return t;
  }

  function renderPlayers() {
    playersLayer.innerHTML = "";
    const ns = "http://www.w3.org/2000/svg";
    state.players.forEach(function (p) {
      const g = document.createElementNS(ns, "g");
      g.setAttribute("data-id", p.id);
      g.setAttribute("class", "player-group " + p.team + (p.hasBall ? " has-ball" : ""));
      g.setAttribute("transform", "translate(" + p.x + "," + p.y + ")");

      const circle = document.createElementNS(ns, "circle");
      circle.setAttribute("data-id", p.id);
      circle.setAttribute("cx", 0);
      circle.setAttribute("cy", 0);
      circle.setAttribute("r", PLAYER_RADIUS);
      circle.setAttribute("class", "player-circle " + p.team + (p.hasBall ? " has-ball" : ""));
      circle.setAttribute("stroke-width", p.hasBall ? "0.8" : "0.4");
      g.appendChild(circle);

      const arrowGroup = document.createElementNS(ns, "g");
      arrowGroup.setAttribute("transform", "rotate(" + rad2deg(p.theta) + ")");
      arrowGroup.setAttribute("class", "player-arrow-wrap");

      const tipX = PLAYER_RADIUS + ARROW_LENGTH;
      const baseX = PLAYER_RADIUS;
      const halfWidth = 0.45;
      const path = document.createElementNS(ns, "path");
      path.setAttribute(
        "d",
        "M " + tipX + " 0 L " + baseX + " " + -halfWidth + " L " + baseX + " " + halfWidth + " Z"
      );
      path.setAttribute("class", "player-arrow");
      arrowGroup.appendChild(path);
      g.appendChild(arrowGroup);

      playersLayer.appendChild(g);
    });
  }

  function setBallOwner(playerId) {
    state.players.forEach(function (p) {
      p.hasBall = p.id === playerId;
    });
    renderPlayers();
    updateThetaUI();
    requestEpvUpdate();
  }

  function buildEpvPayload() {
    const owner = getBallOwner();
    const profileEl = document.getElementById("ball-owner-profile");
    const ballOwnerProfileValue = profileEl ? profileEl.value : "average";
    return {
      frame: 0,
      possessionTeam: owner ? owner.team : "home",
      ballOwnerId: owner ? owner.id : "",
      players: state.players.map(function (p) {
        const out = {
          id: p.id,
          team: p.team,
          x: p.x,
          y: p.y,
          theta: p.theta,
          hasBall: p.hasBall,
        };
        var perDot = state.playerProfileIds[p.id];
        if (perDot != null && perDot !== "average") {
          out.profile_id = perDot;
        } else if (p.hasBall && ballOwnerProfileValue && ballOwnerProfileValue !== "average") {
          out.profile_id = ballOwnerProfileValue;
        }
        return out;
      }),
      ballOwnerProfile: ballOwnerProfileValue === "average" ? "average" : "average",
    };
  }

  function displayEpv(data) {
    epvValueEl.textContent = data.epv != null ? data.epv.toFixed(3) : "—";
    bestActionEl.textContent = data.best_action != null ? data.best_action : "—";
    qShootEl.textContent = data.q_shoot != null ? data.q_shoot.toFixed(3) : "—";
    qPassEl.textContent = data.q_pass != null ? data.q_pass.toFixed(3) : "—";
    qDribbleEl.textContent = data.q_dribble != null ? data.q_dribble.toFixed(3) : "—";
    if (bestActionReasonEl) {
      bestActionReasonEl.textContent = data.best_action_reason != null ? data.best_action_reason : "—";
    }
    if (epvActionDetailEl) {
      let line = "";
      if (data.best_action === "pass" && (data.chosen_receiver_id != null || data.chosen_pass_risk != null)) {
        line = "Receiver: " + (data.chosen_receiver_id != null ? data.chosen_receiver_id : "—") +
          ", risk: " + (data.chosen_pass_risk != null ? data.chosen_pass_risk.toFixed(2) : "—");
      } else if (data.best_action === "shoot" && data.explain && data.explain.shoot) {
        var s = data.explain.shoot;
        line = "Pressure: " + (s.shot_pressure_multiplier != null ? s.shot_pressure_multiplier.toFixed(2) : "—") +
          ", blocked: " + (s.shot_blocked != null ? s.shot_blocked : "—");
      } else if (data.best_action === "dribble" && data.explain && data.explain.dribble) {
        var d = data.explain.dribble;
        line = "Open space: " + (d.dribble_open_space_m != null ? d.dribble_open_space_m.toFixed(1) + " m" : "—");
      }
      epvActionDetailEl.textContent = line;
    }
    if (epvProfileInfoEl) {
      const prof = data.explain && data.explain.profile;
      if (prof && (prof.display_name != null || (prof.position != null && prof.position !== "") || prof.overall_individuality_score != null)) {
        const parts = [];
        if (prof.display_name != null && prof.display_name !== "") parts.push(prof.display_name);
        if (prof.position != null && prof.position !== "") parts.push("Position: " + prof.position);
        if (prof.overall_individuality_score != null) parts.push("Individuality: " + prof.overall_individuality_score.toFixed(2));
        epvProfileInfoEl.textContent = parts.join(" · ");
        epvProfileInfoEl.style.display = "";
      } else {
        epvProfileInfoEl.textContent = "";
        epvProfileInfoEl.style.display = "none";
      }
    }
    renderPassViz(data);
  }

  function getGoalCenterForTeam(team) {
    if (team === "away") return { x: 0, y: PITCH_HEIGHT / 2 };
    return { x: PITCH_WIDTH, y: PITCH_HEIGHT / 2 };
  }

  function renderPassViz(data) {
    if (!passVizLayer) return;
    passVizLayer.innerHTML = "";
    const ns = "http://www.w3.org/2000/svg";
    const owner = getBallOwner();

    if (owner != null && data.best_action_target != null) {
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", owner.x);
      line.setAttribute("y1", owner.y);
      line.setAttribute("x2", data.best_action_target.x);
      line.setAttribute("y2", data.best_action_target.y);
      line.setAttribute("class", "pass-arrow-line");
      line.setAttribute("marker-end", "url(#pass-arrowhead)");
      passVizLayer.appendChild(line);
    }
    if (data.pass_candidates != null && data.pass_candidates.length > 0) {
      data.pass_candidates.forEach(function (c, i) {
        const circle = document.createElementNS(ns, "circle");
        circle.setAttribute("cx", c.x);
        circle.setAttribute("cy", c.y);
        circle.setAttribute("r", 2.2);
        circle.setAttribute("class", "pass-candidate-circle");
        passVizLayer.appendChild(circle);
      });
    }
  }

  function requestEpvUpdate() {
    if (state.epvDebounceTimer) clearTimeout(state.epvDebounceTimer);
    state.epvDebounceTimer = setTimeout(function () {
      state.epvDebounceTimer = null;
      fetchEpv();
    }, EPV_DEBOUNCE_MS);
  }

  function fetchEpv() {
    epvStatusEl.classList.remove("hidden");
    const payload = buildEpvPayload();
    fetch(API_BASE + "/api/epv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (res) {
        if (!res.ok) throw new Error(res.statusText);
        return res.json();
      })
      .then(function (data) {
        displayEpv(data);
      })
      .catch(function (err) {
        displayEpv({});
        epvValueEl.textContent = "Error: " + (err.message || "Failed to fetch");
      })
      .finally(function () {
        epvStatusEl.classList.add("hidden");
      });
  }

  function updateThetaUI() {
    const owner = getBallOwner();
    if (!owner) {
      thetaHint.classList.remove("hidden");
      thetaControls.classList.add("hidden");
      return;
    }
    thetaHint.classList.add("hidden");
    thetaControls.classList.remove("hidden");
    const deg = rad2deg(owner.theta);
    const sliderVal = Math.round(owner.theta * 100);
    thetaSlider.value = Math.max(-314, Math.min(314, sliderVal));
    thetaValue.textContent = owner.theta.toFixed(2);
  }

  function getPlayerAt(x, y) {
    let found = null;
    let minDist = PLAYER_RADIUS + 0.5;
    state.players.forEach(function (p) {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d <= minDist) {
        minDist = d;
        found = p;
      }
    });
    return found;
  }

  renderPlayers();
  updateThetaUI();
  requestEpvUpdate();

  document.getElementById("reset-all").addEventListener("click", function () {
    resetToFormation(createInitialPlayers);
  });
  document.getElementById("reset-433").addEventListener("click", function () {
    resetToFormation(get433Players);
  });
  document.getElementById("reset-442").addEventListener("click", function () {
    resetToFormation(get442Players);
  });

  (function () {
    const profileEl = document.getElementById("ball-owner-profile");
    if (profileEl) {
      profileEl.addEventListener("change", function () {
        requestEpvUpdate();
      });
    }
    loadPlayerProfiles();
  })();

  thetaSlider.addEventListener("input", function () {
    const owner = getBallOwner();
    if (!owner) return;
    owner.theta = (parseInt(thetaSlider.value, 10) / 100);
    owner.theta = normalizeTheta(owner.theta);
    thetaValue.textContent = owner.theta.toFixed(2);
    renderPlayers();
    requestEpvUpdate();
  });

  thetaLeftBtn.addEventListener("click", function () {
    const owner = getBallOwner();
    if (!owner) return;
    owner.theta = normalizeTheta(owner.theta - THETA_STEP);
    thetaSlider.value = Math.round(owner.theta * 100);
    thetaValue.textContent = owner.theta.toFixed(2);
    renderPlayers();
    requestEpvUpdate();
  });

  thetaRightBtn.addEventListener("click", function () {
    const owner = getBallOwner();
    if (!owner) return;
    owner.theta = normalizeTheta(owner.theta + THETA_STEP);
    thetaSlider.value = Math.round(owner.theta * 100);
    thetaValue.textContent = owner.theta.toFixed(2);
    renderPlayers();
    requestEpvUpdate();
  });

  pitchEl.addEventListener("mousedown", function (e) {
    const { x, y } = clientToPitch(e.clientX, e.clientY);
    const player = getPlayerAt(x, y);
    if (!player && state.profilePopoverPlayerId) {
      hideProfilePopover();
    }
    if (!player) return;
    e.preventDefault();
    state.drag = {
      player,
      startX: e.clientX,
      startY: e.clientY,
      startPX: player.x,
      startPY: player.y,
      moved: false,
    };
  });

  window.addEventListener("mousemove", function (e) {
    if (!state.drag) return;
    state.drag.moved = true;
    const { x, y } = clientToPitch(e.clientX, e.clientY);
    const clamped = clampToPitch(x, y);
    state.drag.player.x = clamped.x;
    state.drag.player.y = clamped.y;
    renderPlayers();
  });

  window.addEventListener("mouseup", function (e) {
    if (!state.drag) return;
    const drag = state.drag;
    state.drag = null;
    const dx = e.clientX - drag.startX;
    const dy = e.clientY - drag.startY;
    const movedEnough = Math.hypot(dx, dy) > 4;
    if (!movedEnough) {
      setBallOwner(drag.player.id);
      showProfilePopover(drag.player);
    } else {
      requestEpvUpdate();
    }
  });

  setupProfilePopoverListeners();
})();
