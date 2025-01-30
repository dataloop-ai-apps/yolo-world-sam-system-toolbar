import datetime
import json
import os
import dtlpy as dl


def clean():
    dpk = dl.dpks.get(dpk_name="global-yoloworld-sam-toolbar")

    apps_filters = dl.Filters(field="dpkName", values=dpk.name, resource="apps")
    for app in dl.apps.list(filters=apps_filters).all():
        print(app.name, app.project.name)

        models_filters = dl.Filters(field="app.id", values=app.id, resource="models")
        for model in dl.models.list(filters=models_filters).all():
            print(model.id, model.name, model.creator, model.project.name)
            model.delete()

        app.uninstall()

    _ = [i.delete() for i in list(dpk.revisions.all())]


def publish_and_install(project: dl.Project, manifest):
    env = dl.environment()
    app_name = manifest["name"]
    app_version = manifest["version"]
    print(
        f"Publishing and installing {app_name} {app_version} to project {project.name} in {env}"
    )

    dpk = dl.Dpk.from_json(manifest)
    dpk.codebase = project.codebases.pack(
        directory=os.getcwd(),
        name=dpk.display_name,
        extension="dpk",
        ignore_directories=["artifacts", "venv", "repos"],
        ignore_max_file_size=True,
    )
    # publish dpk to app store
    dpk = project.dpks.publish(dpk=dpk)

    print(
        f"published successfully! dpk name: {dpk.name}, version: {dpk.version}, dpk id: {dpk.id}"
    )
    try:
        app = project.apps.get(app_name=dpk.display_name)
        print("already installed, updating...")
        app.dpk_version = dpk.version
        app.update()
        print(f"update done. app id: {app.id}")
    except dl.exceptions.NotFound:
        print("installing ..")

        app = project.apps.install(
            dpk=dpk, app_name=dpk.display_name, scope=dl.AppScope.SYSTEM
        )
        print(f"installed! app id: {app.id}")
    print("Done!")


def logs():
    s = dl.services.get(service_id="6695258222dfc5793003663c")
    s.log(
        system=True,
        start=datetime.datetime(year=2024, month=12, day=11, hour=0).isoformat(),
        follow=True,
    )


if __name__ == "__main__":
    dl.setenv("rc")
    project = dl.projects.get(project_name="DataloopTasks")
    with open("dataloop.json") as f:
        manifest = json.load(f)
    publish_and_install(manifest=manifest, project=project)
