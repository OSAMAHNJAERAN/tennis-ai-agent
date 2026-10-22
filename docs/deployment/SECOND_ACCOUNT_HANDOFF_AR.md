# فتح المشروع كاملًا من الحساب الثاني

الكود الأحدث موجود على فرع `codex/yolo26-platform`. ملفات المشروع الكبيرة
محفوظة مع نفس المستودع في إصدار
[project-snapshot-2026-10-22](https://github.com/OSAMAHNJAERAN/tennis-ai-agent/releases/tag/project-snapshot-2026-10-22).
بعد نشر الإصدار، يستطيع أي حساب تنزيله لأن المستودع عام.

## تنزيل النسخة كاملة

ثبّت Git وGit LFS وGitHub CLI وPython 3.11 أو أحدث، ثم نفّذ في PowerShell:

```powershell
git clone --branch codex/yolo26-platform https://github.com/OSAMAHNJAERAN/tennis-ai-agent.git
cd tennis-ai-agent
git lfs install --local
git lfs pull
gh release download project-snapshot-2026-10-22 --repo OSAMAHNJAERAN/tennis-ai-agent --pattern "objects-*.zip" --dir project_snapshot
python scripts/maintenance/restore_project_snapshot.py --destination .
```

نزّل الأجزاء الـ48 كلها. حجم الأجزاء المضغوطة نحو 22.7 GiB، وتحتاج أيضًا
مساحة لفكها واستعادة ملفات المشروع الأصلية. تفحص أداة الاستعادة بصمات
SHA-256 للأجزاء ولمحتوياتها قبل إعادة الملفات لمساراتها الأصلية. إذا وجدت
ملفًا مختلفًا موجودًا مسبقًا، تتوقف بدل استبداله تلقائيًا.

الأرشيف يحفظ 526,806 ملفًا إضافيًا، ومنها النماذج والبيانات والفيديوهات
ومخرجات التدريب والتحقق. الكود والوثائق والإعدادات والاختبارات موجودة
بمساراتها المعتادة في Git. تفاصيل الحصر والبصمات في
[فهرس النسخة](../../project_snapshot/index.json) و[دليل الاستعادة](../../project_snapshot/README.md).

حزم الاعتماد المثبتة والبيئات الافتراضية والكاش وبيانات Git الداخلية
ليست ضمن الأرشيف. أعد تثبيت الاعتمادات بحسب
[دليل التثبيت](PRODUCT_INSTALLATION.md) و[وصف تشغيل المنتج](../PRODUCT_WORKFLOW.md).
استخدم `.env.example` لإنشاء إعداداتك المحلية عند الحاجة.

## متابعة العمل

افتح مجلد `tennis-ai-agent` من الحساب الثاني، وابدأ بقراءة `AGENT.md`
و`docs/plans/YOLO26_EXECUTION_PROGRESS.md`. النسخة تحفظ حالة العمل ونتائج
التحقق؛ لا تعني أن جميع أهداف المشروع اكتملت.

الوصول العام يتيح تنزيل الملفات والعمل محليًا. لدفع التعديلات إلى نفس
المستودع، سجّل الدخول بحساب GitHub لديه صلاحية الكتابة؛ ويمكن لحساب آخر
متابعة العمل في fork خاص به وإرسال pull request.
