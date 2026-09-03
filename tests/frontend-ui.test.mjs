import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const stylesheet = fs.readFileSync('web/styles.css', 'utf8');

test('learner workspace uses the approved blue and white palette', () => {
  assert.match(stylesheet, /--paper:#F5F7FF/);
  assert.match(stylesheet, /--surface:#FFFFFF/);
  assert.match(stylesheet, /--accent:#5367E8/);
  assert.match(stylesheet, /--accent-soft:#EEF1FF/);
  assert.match(stylesheet, /\.kind\{color:var\(--technical\)\}/);
});

const sandbox = {};
vm.createContext(sandbox);
const source = fs.existsSync('web/view-model.js')
  ? fs.readFileSync('web/view-model.js', 'utf8')
  : '';
vm.runInContext(source, sandbox);
const view = sandbox.AgentEduView || {};
const required = name => {
  assert.equal(typeof view[name], 'function', `${name} must be exported`);
  return view[name];
};

test('cleanDisplayText removes complete and unclosed reasoning blocks', () => {
  const cleanDisplayText = required('cleanDisplayText');
  assert.equal(cleanDisplayText('<think>secret</think>\n先学习安全规程。'), '先学习安全规程。');
  assert.equal(cleanDisplayText('<think>secret'), '');
});

test('resourcesForKp returns only the selected learning unit', () => {
  const resources = [{kp: 'KP-01'}, {kp: 'KP-02'}, {kp: 'KP-01'}];
  const resourcesForKp = required('resourcesForKp');
  assert.deepEqual(
    JSON.parse(JSON.stringify(resourcesForKp(resources, 'KP-01'))),
    [{kp: 'KP-01'}, {kp: 'KP-01'}],
  );
});

test('feedbackNextAction keeps remediation on the current unit', () => {
  const feedbackNextAction = required('feedbackNextAction');
  assert.deepEqual(
    JSON.parse(JSON.stringify(feedbackNextAction(['KP-01', 'KP-02'], 'KP-01', {action: 'explain_down'}))),
    {kind: 'repeat', label: '重新学习本知识点', targetKp: 'KP-01'},
  );
});

test('feedbackNextAction advances to the next unit after success', () => {
  const feedbackNextAction = required('feedbackNextAction');
  assert.deepEqual(
    JSON.parse(JSON.stringify(feedbackNextAction(['KP-01', 'KP-02'], 'KP-01', {action: 'advance'}))),
    {kind: 'next', label: '继续下一知识点', targetKp: 'KP-02'},
  );
});

test('feedbackNextAction finishes when there is no following unit', () => {
  assert.deepEqual(
    JSON.parse(JSON.stringify(view.feedbackNextAction(['KP-01'], 'KP-01', {action: 'advance'}))),
    {kind: 'summary', label: '查看更新后的学习建议', targetKp: null},
  );
});

test('feedbackNextAction keeps consolidation on the current unit', () => {
  assert.deepEqual(
    JSON.parse(JSON.stringify(view.feedbackNextAction(['KP-01', 'KP-02'], 'KP-01', {action: 'consolidate'}))),
    {kind: 'repeat', label: '重新学习本知识点', targetKp: 'KP-01'},
  );
});

const appSandbox = {window: {AgentEduView: view}};
vm.createContext(appSandbox);
const appSource = fs.existsSync('web/app.js')
  ? fs.readFileSync('web/app.js', 'utf8')
  : '';
vm.runInContext(appSource, appSandbox);
const app = appSandbox.AgentEduApp || {};
const requiredApp = name => {
  assert.equal(typeof app[name], 'function', `${name} must be exported`);
  return app[name];
};

function createInteractiveDom() {
  const focusEvents = [];
  const scrollEvents = [];
  const interactionEvents = [];
  let pathButtons = [];
  let feedbackQuestions = [];
  let resourceGroups = [];

  const decode = value => String(value)
    .replaceAll('&amp;', '&')
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'");
  const textOnly = value => decode(String(value).replace(/<[^>]*>/g, ''));

  const makeElement = id => {
    let html = '';
    let value = '';
    const element = {
      id,
      hidden: true,
      disabled: false,
      textContent: '',
      dataset: {},
      attributes: {},
      options: [],
      classList: {add() {}, remove() {}, toggle() {}},
      setAttribute(name, next) {
        this.attributes[name] = String(next);
        if (name === 'aria-current') this.ariaCurrent = String(next);
      },
      removeAttribute(name) {
        delete this.attributes[name];
        if (name === 'data-on') delete this.dataset.on;
      },
      focus(options) { const event = {id, options}; focusEvents.push(event); interactionEvents.push({kind: 'focus', id}); },
      scrollIntoView(options) { const event = {id, options}; scrollEvents.push(event); interactionEvents.push({kind: 'scroll', id}); },
      append() {},
      appendChild() {},
      insertAdjacentHTML(_position, next) { this.innerHTML += next; },
    };
    Object.defineProperty(element, 'value', {
      get() { return value; },
      set(next) { value = next == null ? '' : String(next); },
    });
    Object.defineProperty(element, 'innerHTML', {
      get() { return html; },
      set(next) {
        html = String(next);
        if (id === 'fbKp') {
          element.options = [...html.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)]
            .map(match => ({value: decode(match[1]), textContent: decode(match[2])}));
          value = element.options[0]?.value || '';
        }
        if (id === 'learningPath') {
          pathButtons = [...html.matchAll(/<button[^>]*data-learning-kp="([^"]*)"[^>]*aria-current="([^"]*)"[^>]*>([^<]*)<\/button>/g)]
            .map(match => {
              const button = makeElement('');
              button.dataset.learningKp = decode(match[1]);
              button.ariaCurrent = match[2];
              button.textContent = decode(match[3]);
              return button;
            });
        }
        if (id === 'quiz') {
          feedbackQuestions = [...html.matchAll(/<div class="feedback-question" data-index="(\d+)">([\s\S]*?)(?=<div class="feedback-question"|$)/g)]
            .map(match => {
              const question = makeElement('');
              question.dataset.index = match[1];
              question.textContent = textOnly(match[2]);
              const explain = {hidden: true};
              const answers = ['1', '0'].map(answerValue => {
                const answer = makeElement('');
                answer.dataset.value = answerValue;
                answer.closest = () => question;
                return answer;
              });
              question.querySelectorAll = selector => selector === '.answer' ? answers : [];
              question.querySelector = selector => selector === '.feedback-explain' ? explain : undefined;
              question.answers = answers;
              return question;
            });
        }
        if (id === 'resources') {
          resourceGroups = [...html.matchAll(/<section class="resource-group" data-resource-group="([^"]+)">([\s\S]*?)(?=<section class="resource-group"|$)/g)]
            .map(match => ({dataset: {resourceGroup: match[1]}, textContent: textOnly(match[2])}));
        }
        if (id === 'decisionPanel') {
          const match = html.match(/<button[^>]*id="continueLearning"[^>]*>([^<]*)<\/button>/);
          if (match) elements.continueLearning.textContent = decode(match[1]);
        }
      },
    });
    return element;
  };

  const ids = [
    'submitFb', 'verdict', 'decisionPanel', 'continueLearning', 'workflowProgress',
    'timeline', 'ecount', 'rcount', 'resources', 'learningPath', 'learningContent',
    'learningPlan', 'fbKp', 'quiz', 'chartFit', 'chartPath', 'uiAlert',
  ];
  const elements = Object.fromEntries(ids.map(id => [id, makeElement(id)]));
  const document = {
    querySelector(selector) { return elements[selector.slice(1)]; },
    querySelectorAll(selector) {
      if (selector === '[data-learning-kp]') return pathButtons;
      if (selector === '.feedback-question') return feedbackQuestions;
      if (selector === '.answer') return feedbackQuestions.flatMap(question => question.answers);
      if (selector === '[data-resource-group]') return resourceGroups;
      return [];
    },
    createElementNS() { return makeElement(''); },
  };
  return {
    document,
    elements,
    focusEvents,
    scrollEvents,
    interactionEvents,
    get feedbackQuestions() { return feedbackQuestions; },
    get resourceGroups() { return resourceGroups; },
  };
}

test('resourceBody preserves headings, evidence, numbered steps, and bullets', () => {
  const resourceBody = requiredApp('resourceBody');
  const html = resourceBody({
    kind: 'guide',
    body: '# 安全准备\n## 操作步骤\n> 依据 KB-01\n1. 检查急停\n- 记录结果',
  });
  assert.match(html, /<h3>安全准备<\/h3>/);
  assert.match(html, /<h4>操作步骤<\/h4>/);
  assert.match(html, /class="rsource">依据 KB-01/);
  assert.match(html, /class="rpoint">1\. 检查急停/);
  assert.match(html, /class="rbullet">记录结果/);
});

test('chart models retain scales, learner fit, and path completion semantics', () => {
  const fitChartModel = requiredApp('fitChartModel');
  const pathChartModel = requiredApp('pathChartModel');
  const session = {
    resources: [{kp: 'KP-01', difficulty: 3}, {kp: 'KP-01', difficulty: 4}],
    diagnosis: {mastery: [{kp: 'KP-01', score: 0.5}]},
    path: ['KP-01'], path_names: ['安全规程'], kp_index: { 'KP-01': {name: '安全规程'} },
  };
  assert.deepEqual(JSON.parse(JSON.stringify(fitChartModel(session).levels)), [1, 2, 3, 4, 5]);
  assert.deepEqual(JSON.parse(JSON.stringify(fitChartModel(session).items[0])), {
    kp: 'KP-01', name: '安全规程', learnerLevel: 3, windowTop: 5, difficulties: [3, 4],
  });
  assert.deepEqual(JSON.parse(JSON.stringify(pathChartModel(session)[0])), {
    name: '安全规程', completed: true, resourceCount: 2, difficulty: 3,
  });
});

test('resource count refreshes from the feedback-updated session', () => {
  const renderResourceCount = requiredApp('renderResourceCount');
  const heading = {textContent: ''};
  const count = renderResourceCount({resources: [{}, {}, {}]}, heading);
  assert.equal(count, 3);
  assert.equal(heading.textContent, 3);
});

test('intake and material failures announce errors through the alert region', async () => {
  const parseIntake = requiredApp('parseIntake');
  const stageMaterial = requiredApp('stageMaterial');
  const elements = {
    '#parseIntake': {disabled: false, textContent: ''},
    '#intakeText': {value: '测试学习经历'},
    '#intakeSummary': {hidden: true, textContent: ''},
    '#startInterview': {disabled: true},
    '#materialFile': {files: []},
    '#stageResult': {hidden: true, textContent: '', classList: {remove() {}}},
    '#uiAlert': {hidden: true, textContent: ''},
  };
  appSandbox.document = {querySelector: selector => elements[selector]};
  appSandbox.fetch = async () => ({ok: false, json: async () => ({error: '服务不可用'})});

  await parseIntake();
  assert.equal(elements['#uiAlert'].hidden, false);
  assert.match(elements['#uiAlert'].textContent, /无法解析：服务不可用/);

  elements['#uiAlert'].hidden = true;
  await stageMaterial();
  assert.equal(elements['#uiAlert'].hidden, false);
  assert.match(elements['#uiAlert'].textContent, /请先选择要提交的资料/);
});

test('gap chart exposes each learner mastery score as visible text', () => {
  const gapChartModel = requiredApp('gapChartModel');
  const drawGaps = requiredApp('drawGaps');
  const item = gapChartModel({
    diagnosis: {gaps: ['KP-01'], mastery: [{kp: 'KP-01', name: '安全规程', score: 0.5, correct: 2, asked: 4}]},
  })[0];
  assert.equal(item.scoreLabel, '掌握度 50.0%');

  const svg = {innerHTML: '', nodes: [], append(...nodes) { this.nodes.push(...nodes); }, appendChild(node) { this.nodes.push(node); }};
  appSandbox.document = {
    querySelector: selector => selector === '#chartGaps' ? svg : undefined,
    createElementNS: () => ({setAttribute() {}, textContent: ''}),
  };
  requiredApp('setSession')({diagnosis: {gaps: ['KP-01'], mastery: [{kp: 'KP-01', name: '安全规程', score: 0.5, correct: 2, asked: 4}]}});
  drawGaps();
  assert.ok(svg.nodes.some(node => node.textContent === '掌握度 50.0% · 2/4'));
});

test('submitFb refreshes the rendered resource count after feedback returns new resources', async () => {
  const submitFb = requiredApp('submitFb');
  const setSession = requiredApp('setSession');
  const element = () => ({hidden: true, textContent: '', innerHTML: '', disabled: false, value: 'KP-01', focus() {}, scrollIntoView() {}, append() {}, appendChild() {}, insertAdjacentHTML() {}});
  const elements = {
    '#submitFb': element(), '#verdict': element(), '#decisionPanel': element(), '#continueLearning': element(), '#workflowProgress': element(),
    '#timeline': element(), '#rcount': element(), '#resources': element(), '#learningPath': element(),
    '#learningContent': element(), '#fbKp': element(), '#quiz': element(), '#chartFit': element(), '#chartPath': element(), '#uiAlert': element(),
  };
  appSandbox.document = {
    querySelector: selector => elements[selector],
    querySelectorAll: selector => selector === '.feedback-question' ? [{dataset: {pick: '1'}}] : [],
    createElementNS: () => ({setAttribute() {}, textContent: '', appendChild() {}}),
  };
  const before = {session_id: 'S-1', events: [], resources: [], diagnosis: {mastery: [], gaps: []}, path: ['KP-01'], path_names: ['安全规程'], kp_index: {'KP-01': {name: '安全规程'}}};
  const after = {...before, resources: [{kp: 'KP-01', kind: 'guide', title: '新增资料', difficulty: 2, claims: [], body: '内容'}], decision: {action: 'advance', reason: '已掌握'}, feedback_result: [{correct: true, answer: true, explain: '依据测试资料'}]};
  setSession(before);
  let submitted;
  appSandbox.fetch = async (_url, options) => {
    submitted = JSON.parse(options.body);
    return {ok: true, json: async () => after};
  };

  await submitFb();
  assert.equal(elements['#rcount'].textContent, 1);
  assert.deepEqual(submitted.choices, [true]);
  assert.equal('answers' in submitted, false);
});

test('feedback append rebuilds the selector and continue keeps resources and quiz synchronized', async () => {
  const dom = createInteractiveDom();
  appSandbox.document = dom.document;
  appSandbox.matchMedia = () => ({matches: false});
  const before = {
    session_id: 'S-append', events: [], kb: {},
    diagnosis: {mastery: [{kp: 'KP-01', score: 0.5}], gaps: ['KP-01']},
    path: ['KP-01'], path_names: ['安全规程'], kp_index: {'KP-01': {name: '安全规程'}},
    resources: [
      {kp: 'KP-01', kind: 'lecture', title: '基础讲解', difficulty: 2, claims: [], body: '讲解'},
      {kp: 'KP-01', kind: 'quiz', title: '安全自测', difficulty: 2, claims: [], items: [{stem: '旧题'}]},
    ],
  };
  const after = {
    ...before,
    path: ['KP-01', 'KP-02'], path_names: ['安全规程', '故障案例'],
    kp_index: {'KP-01': {name: '安全规程'}, 'KP-02': {name: '故障案例'}},
    resources: [
      ...before.resources,
      {kp: 'KP-01', kind: 'lecture_simplified', title: '新增浅显讲解', difficulty: 1, claims: [], body: '补充'},
      {kp: 'KP-02', kind: 'lecture', title: '故障讲解', difficulty: 3, claims: [], body: '讲解'},
      {kp: 'KP-02', kind: 'case', title: '故障案例', difficulty: 3, claims: [], body: '案例'},
      {kp: 'KP-02', kind: 'quiz', title: '故障自测', difficulty: 3, claims: [], items: [{stem: '新题'}]},
    ],
    decision: {action: 'advance', reason: '可以进入下一知识点'},
    feedback_result: [{correct: true, answer: true, explain: '回答正确'}],
  };

  requiredApp('setSession')(before);
  requiredApp('selectLearningUnit')('KP-01');
  dom.feedbackQuestions[0].answers[0].onclick();
  assert.equal(dom.feedbackQuestions[0].dataset.pick, '1');
  appSandbox.fetch = async () => ({ok: true, json: async () => after});

  await requiredApp('submitFb')();

  assert.deepEqual(dom.elements.fbKp.options.map(option => option.value), ['KP-01', 'KP-02']);
  assert.equal(dom.elements.fbKp.value, 'KP-01');
  assert.equal(dom.feedbackQuestions[0].dataset.pick, undefined, 'feedback refresh clears old picks');
  const lectureGroup = dom.resourceGroups.find(group => group.dataset.resourceGroup === 'lecture');
  assert.match(lectureGroup.textContent, /课程讲解/);
  assert.match(lectureGroup.textContent, /基础讲解/);
  assert.match(lectureGroup.textContent, /新增浅显讲解/);

  dom.feedbackQuestions[0].answers[0].onclick();
  dom.elements.continueLearning.onclick();

  assert.equal(dom.elements.fbKp.value, 'KP-02');
  assert.deepEqual(dom.resourceGroups.map(group => group.dataset.resourceGroup), ['lecture', 'case', 'quiz']);
  assert.match(dom.resourceGroups[1].textContent, /案例练习.*故障案例/);
  assert.match(dom.elements.quiz.innerHTML, /新题/);
  assert.equal(dom.feedbackQuestions[0].dataset.pick, undefined);
  assert.equal(dom.focusEvents.at(-1).id, 'learningContent');
  assert.equal(dom.scrollEvents.at(-1).id, 'learningContent');
  assert.deepEqual(dom.interactionEvents.slice(-2), [
    {kind: 'focus', id: 'learningContent'},
    {kind: 'scroll', id: 'learningContent'},
  ]);

  requiredApp('selectLearningUnit')('KP-01');
  assert.equal(dom.feedbackQuestions[0].dataset.pick, undefined, 'returning to a unit never restores stale picks');
});

test('repeat and summary actions focus before scrolling to their destinations', () => {
  const dom = createInteractiveDom();
  appSandbox.document = dom.document;
  appSandbox.matchMedia = () => ({matches: false});
  requiredApp('setSession')({
    path: ['KP-01'], path_names: ['安全规程'], kb: {},
    resources: [{kp: 'KP-01', kind: 'quiz', title: '安全自测', difficulty: 2, claims: [], items: [{stem: '题目'}]}],
  });
  requiredApp('selectLearningUnit')('KP-01');

  requiredApp('renderDecision')({action: 'consolidate', reason: '需要巩固'}, [{correct: true, answer: true, explain: ''}]);
  dom.elements.continueLearning.onclick();
  assert.equal(dom.focusEvents.at(-1).id, 'learningContent');
  assert.equal(dom.focusEvents.at(-1).options.preventScroll, true);
  assert.equal(dom.scrollEvents.at(-1).id, 'learningContent');
  assert.deepEqual(dom.interactionEvents.slice(-2), [
    {kind: 'focus', id: 'learningContent'},
    {kind: 'scroll', id: 'learningContent'},
  ]);

  requiredApp('renderDecision')({action: 'advance', reason: '已完成'}, [{correct: true, answer: true, explain: ''}]);
  dom.elements.continueLearning.onclick();
  assert.equal(dom.focusEvents.at(-1).id, 'learningPlan');
  assert.equal(dom.focusEvents.at(-1).options.preventScroll, true);
  assert.equal(dom.scrollEvents.at(-1).id, 'learningPlan');
  assert.deepEqual(dom.interactionEvents.slice(-2), [
    {kind: 'focus', id: 'learningPlan'},
    {kind: 'scroll', id: 'learningPlan'},
  ]);
});
