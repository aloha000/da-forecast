"""Shared 6h model, origin-only observations, full differentiable autoregression."""
import torch
from torch.utils.checkpoint import checkpoint

def rollout_predictions(model,batch,horizon,checkpoint_steps=False):
    if not 1<=horizon<=batch["rollout_time_features"].shape[1]:
        raise ValueError("missing calendar features for requested horizon")
    history=batch["era5"]
    for step in range(horizon):
        # Pass every changing input explicitly; no late-bound closure over step.
        features=batch["rollout_time_features"][:,step]
        def run(x,static,calendar,obs,mask,obs_static,enabled):
            return model(x,static,calendar,obs,mask,obs_static,obs_enabled=enabled)
        args=(history,batch["era_static"],features,
              batch["obs_ztd"] if step==0 else None,
              batch["obs_mask"] if step==0 else None,
              batch["obs_static"] if step==0 else None,step==0)
        if checkpoint_steps and torch.is_grad_enabled():
            prediction=checkpoint(run,*args,use_reentrant=False)
        else:
            prediction=run(*args)
        yield prediction
        # Never detach and never replace prediction with a future ERA label.
        if step+1<horizon:
            history=torch.stack((history[:,-1],prediction),dim=1)

def rollout_loss(model,batch,horizon,lat,config):
    from train import weighted_losses
    total=0.
    per_lead=[]
    for j,pred in enumerate(rollout_predictions(model,batch,horizon,
                               config.get("rollout_checkpoint",True))):
        value,_,_=weighted_losses(pred.float(),batch["targets"][:,j],
            batch["target_masks"][:,j],batch["obs_static"],lat,
            config.get("halo_weight",.5),config.get("halo_radius",3),
            config.get("loss_mode","global_only"))
        total=total+value/horizon
        per_lead.append(value.detach())
    return total,torch.stack(per_lead)
