// UI-only directory draft. No filesystem, network, URL, or persistence access.
const MAX_PATH_LENGTH = 2048;

/** showDialog/closeDialog receive the dialog element; all paths stay in memory. */
export function initCheckpointDirectory({ getCheckpoint, onNotice, onDraftChange, showDialog, closeDialog }) {
  const required = id => {
    const element = document.getElementById(id);
    if (!element) throw new Error(`Missing checkpoint directory control: ${id}`);
    return element;
  };
  const current = required('checkpoint-dir-current');
  const draft = required('checkpoint-dir-draft');
  const open = required('checkpoint-dir-open');
  const choose = required('checkpoint-dir-choose');
  const dialog = required('checkpoint-dir-dialog');
  const input = required('checkpoint-dir-input');
  const apply = required('checkpoint-dir-apply');
  let currentPath = null;
  let draftPath = null;
  let destroyed = false;
  const originalMaxLength = input.maxLength;
  input.maxLength = MAX_PATH_LENGTH;

  function clearInvalid() {
    input.setCustomValidity('');
    input.removeAttribute('aria-invalid');
  }

  function update() {
    if (destroyed) return;
    const checkpoint = getCheckpoint();
    const alias = checkpoint ? String(checkpoint.alias ?? '').trim() || 'current' : null;
    currentPath = checkpoint ? `checkpoints/${alias}` : null;
    current.textContent = currentPath ?? '尚无已保存的 checkpoint';
    open.disabled = currentPath === null;
    draft.textContent = draftPath ? `已选择：${draftPath}` : currentPath ? '使用当前保存目录' : '尚未选择目录';
  }

  function handleOpen(event) {
    event.preventDefault();
    update();
    if (currentPath === null) return;
    onNotice(`保存位置：${currentPath}`);
  }

  function handleChoose(event) {
    event.preventDefault();
    update();
    input.value = draftPath ?? currentPath ?? 'checkpoints/';
    clearInvalid();
    showDialog(dialog);
  }

  function handleApply(event) {
    event.preventDefault();
    const path = input.value.trim();
    const error = !path ? '请输入目录路径。' : path.length > MAX_PATH_LENGTH ? '目录路径最多允许 2048 个字符。' : '';
    if (error) {
      input.setCustomValidity(error);
      input.setAttribute('aria-invalid', 'true');
      input.reportValidity();
      return;
    }
    clearInvalid();
    draftPath = path;
    input.value = path;
    update();
    onDraftChange(path);
    onNotice('目录选择已更新。');
    closeDialog(dialog);
  }

  function handleInputKey(event) {
    if (event.key === 'Enter' && !event.isComposing) handleApply(event);
  }

  open.addEventListener('click', handleOpen);
  choose.addEventListener('click', handleChoose);
  apply.addEventListener('click', handleApply);
  input.addEventListener('input', clearInvalid);
  input.addEventListener('keydown', handleInputKey);
  update();

  return {
    update,
    getDraftPath: () => draftPath,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      open.removeEventListener('click', handleOpen);
      choose.removeEventListener('click', handleChoose);
      apply.removeEventListener('click', handleApply);
      input.removeEventListener('input', clearInvalid);
      input.removeEventListener('keydown', handleInputKey);
      input.maxLength = originalMaxLength;
      clearInvalid();
    },
  };
}
